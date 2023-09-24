extern crate sudoku;
#[macro_use]
extern crate clap;

use sudoku::Sudoku;

extern crate rayon;
use rayon::prelude::*;

use clap::{Arg, ArgAction, Command};
use std::io::{self, Read, Write};
use sudoku::errors::LineParseError;

enum ActionsKind {
    Single(SingleThreaded),
    Multi(MultiThreaded),
}

impl Actions for ActionsKind {
    fn solve_and_print(&self, input: &str, path: Option<&std::path::Path>) {
        use ActionsKind::*;
        match self {
            Single(s) => s.solve_and_print(input, path),
            Multi(m) => m.solve_and_print(input, path),
        }
    }

    fn solve_and_print_stats(&self, input: &str, path: Option<&std::path::Path>, stats: bool) {
        use ActionsKind::*;
        match self {
            Single(s) => s.solve_and_print_stats(input, path, stats),
            Multi(m) => m.solve_and_print_stats(input, path, stats),
        }
    }

    fn gen_sudokus(&self, count: usize, generator: impl Fn() -> Sudoku + Sync, print_blocks: bool) {
        use ActionsKind::*;
        match self {
            Single(s) => s.gen_sudokus(count, generator, print_blocks),
            Multi(m) => m.gen_sudokus(count, generator, print_blocks),
        }
    }
}

struct SingleThreaded;
struct MultiThreaded;

trait Actions {
    fn solve_and_print(&self, input: &str, path: Option<&std::path::Path>);
    fn solve_and_print_stats(&self, input: &str, path: Option<&std::path::Path>, stats: bool);
    fn gen_sudokus(&self, count: usize, generator: impl Fn() -> Sudoku + Sync, print_blocks: bool);
}

impl Actions for SingleThreaded {
    fn solve_and_print(&self, input: &str, path: Option<&std::path::Path>) {
        let sudokus = input
            .lines()
            .map(sudoku::Sudoku::from_str_line)
            .map(|maybe_sudoku| maybe_sudoku.map(|sudoku| sudoku.solution().ok_or(sudoku)));

        _print(sudokus, path);
    }

    fn solve_and_print_stats(&self, input: &str, path: Option<&std::path::Path>, stats: bool) {
        let sudokus = input
            .lines()
            .map(sudoku::Sudoku::from_str_line)
            .map(|maybe_sudoku| maybe_sudoku.map(|sudoku| sudoku.solutions_count_up_to(2)));

        let time = Time::DoMeasure(stats);

        _print_stats(sudokus, path, stats, time);
    }

    fn gen_sudokus(&self, count: usize, generator: impl Fn() -> Sudoku, print_blocks: bool) {
        let stdout = io::stdout();
        let mut lock = stdout.lock();
        for _ in 0..count {
            if print_blocks {
                let _ = writeln!(lock, "{}\n", generator().display_block());
            } else {
                let _ = writeln!(lock, "{}", generator());
            }
        }
    }
}

impl Actions for MultiThreaded {
    fn solve_and_print(&self, input: &str, path: Option<&std::path::Path>) {
        let sudokus = input
            .par_lines()
            .map(sudoku::Sudoku::from_str_line)
            .map(|maybe_sudoku| maybe_sudoku.map(|sudoku| sudoku.solution().ok_or(sudoku)))
            .collect::<Vec<_>>()
            .into_iter();

        _print(sudokus, path);
    }

    fn solve_and_print_stats(&self, input: &str, path: Option<&std::path::Path>, stats: bool) {
        use std::time::Instant;

        let start = match stats {
            true => Some(Instant::now()),
            false => None,
        };
        let sudokus = input
            .par_lines()
            .map(sudoku::Sudoku::from_str_line)
            .map(|maybe_sudoku| maybe_sudoku.map(|sudoku| sudoku.solutions_count_up_to(2)))
            .collect::<Vec<_>>()
            .into_iter();
        let duration = start.map(|start| Instant::now() - start);
        let time = match duration {
            Some(duration) => Time::Measured(duration),
            None => Time::DoMeasure(false),
        };

        _print_stats(sudokus, path, stats, time);
    }

    fn gen_sudokus(&self, count: usize, generator: impl Fn() -> Sudoku + Sync, print_blocks: bool) {
        (0..count).into_par_iter().for_each(|_| {
            if print_blocks {
                println!("{}\n", generator().display_block());
            } else {
                println!("{}", generator());
            }
        });
    }
}

enum Time {
    Measured(std::time::Duration),
    DoMeasure(bool),
}

fn _print<I: Iterator<Item = Result<Result<Sudoku, Sudoku>, LineParseError>>>(
    sudokus: I,
    _path: Option<&std::path::Path>,
) {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();

    for sud in sudokus {
        match sud {
            Ok(Ok(solution)) => {
                let _ = writeln!(lock, "{}", solution);
            }
            Ok(Err(original)) => {
                let _ = writeln!(lock, "{} no unique solution", original);
            }
            Err(e) => {
                let _ = writeln!(lock, "invalid sudoku: {}", e);
            }
        };
    }
}

fn _print_stats<I: Iterator<Item = Result<usize, LineParseError>>>(
    sudokus: I,
    path: Option<&std::path::Path>,
    count: bool,
    time: Time,
) {
    use std::time::Instant;
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let (mut n_solved, mut n_invalid, mut n_non_unique) = (0, 0, 0);

    let start = match time {
        Time::DoMeasure(true) => Some(Instant::now()),
        _ => None,
    };

    for sud in sudokus {
        match sud {
            Ok(0) => n_invalid += 1,
            Ok(1) => n_solved += 1,
            Ok(_) => n_non_unique += 1,
            Err(e) => {
                let _ = eprintln!("invalid sudoku: {}", e);
                n_invalid += 1;
            }
        };
    }

    let duration = start.map(|start| Instant::now() - start).or(match time {
        Time::Measured(duration) => Some(duration),
        _ => None,
    });

    let total = n_solved + n_invalid + n_non_unique;

    if count {
        let _ = write!(
            lock,
            "{:>9} {:>9} {:>9} {:>9} ",
            total, n_solved, n_non_unique, n_invalid
        );
    }
    if let Some(time_taken) = duration {
        let seconds = time_taken.as_secs() as f64 + time_taken.subsec_nanos() as f64 * 1e-9;
        let solving_rate = total as f64 / seconds;
        let _ = write!(lock, "{:>10.3} {:>10.0} ", seconds, solving_rate);
    }

    if let Some(path) = path {
        let _ = write!(lock, "{}", path.display());
    }
    let _ = write!(lock, "\n");
}

fn read_stdin(buffer: &mut String) {
    let stdin = io::stdin();
    let mut lock = stdin.lock();
    let _ = lock.read_to_string(buffer);
}

fn read_sudokus_and_execute<F>(matches: &clap::ArgMatches, mut callback: F)
where
    F: FnMut(Option<&std::path::Path>, &str),
{
    let mut sudoku_buffer = String::new();

    if let Some(filenames) = matches.values_of("sudokus_file") {
        for filename in filenames {
            let path = std::path::Path::new(filename);

            let mut file = match std::fs::File::open(path) {
                Ok(f) => f,
                Err(e) => {
                    println!("Could not open file: {}", e);
                    return;
                }
            };
            let _ = file.read_to_string(&mut sudoku_buffer);
            callback(Some(path), &sudoku_buffer);
            sudoku_buffer.clear();
        }
    } else {
        read_stdin(&mut sudoku_buffer);
        callback(None, &sudoku_buffer);
    }
}

fn actions_object(mut parallel: bool, no_parallel: bool) -> ActionsKind {
    if !parallel && !no_parallel {
        parallel = cfg!(feature = "parallel_by_default");
    }
    match parallel {
        true => ActionsKind::Multi(MultiThreaded),
        false => ActionsKind::Single(SingleThreaded),
    }
}

fn main() {
    let mut app = Command::new("rustoku")
        .version(crate_version!())
        .about("Solves and generates sudokus")
        .subcommand(
            Command::new("solve")
                .about("Solve sudokus")
                .arg(
                    Arg::new("sudokus_file")
                        .value_name("FILE")
                        .action(ArgAction::Append)
                )
                .arg(
                    Arg::new("statistics")
                        .long("stat")
                        .short('s')
                        .help("do not print solutions, but categorize sudokus by solution count (1, >1, 0) and measure solving speed")
                )
                .arg(
                    Arg::new("parallel")
                        .long("parallel")
                        .short('p')
                        .overrides_with("no-parallel")
                        .help("Use multiple threads")
                )
                .arg(
                    Arg::new("no-parallel")
                        .long("no-parallel")
                        .overrides_with("parallel")
                        .help("Do not use multiple threads")
                )
        )
        .subcommand(
            Command::new("generate")
                .about("Generate sudokus")
                .arg(
                    Arg::new("amount")
                        .takes_value(true)
                        .required(true)
                )
                .arg(
                    Arg::new("print-block")
                        .long("print-block")
                        .short('b')
                        .help("print sudokus as blocks")
                )
                .arg(
                    Arg::new("solved")
                        .help("generate solved sudokus")
                        .long("solved")
                )
                .arg(
                    Arg::new("parallel")
                        .long("parallel")
                        .short('p')
                        .overrides_with("no-parallel")
                        .help("Use multiple threads")
                )
                .arg(
                    Arg::new("no-parallel")
                        .long("no-parallel")
                        .overrides_with("parallel")
                        .help("Do not use multiple threads")
                )
        )
        .subcommand(
            Command::new("shuffle")
                .about("Create different, but equivalent sudokus")
                .long_about("Performs symmetry transformations (swapping digits, stacks, bands, rows within bands, columns within stacks and mirroring at the diagonal) to generate random different, but equivalent sudokus")
                .arg(
                    // TODO: Decide on how to unify amount (on generate option) and count
                    Arg::new("count")
                        .takes_value(true)
                        .short('n')
                )
                .arg(
                    Arg::new("sudokus_file")
                        .value_name("FILE")
                        .action(ArgAction::Append)
                )
        )
        .subcommand(
            Command::new("canonicalize")
                .about("Find the lexicographically minimal, equivalent sudoku")
                .long_about("Performs symmetry transformations (swapping digits, stacks, bands, rows within bands, columns within stacks and mirroring at the diagonal) to find the lexicographically minimal, equivalent sudoku.\nThe number of transformations resulting in the same sudoku (automorphisms) is given after each sudoku's canonical version. It is at least 1 (identity transformation) and at most 648.")
                .arg(
                    Arg::new("sudokus_file")
                        .takes_value(true)
                        .value_name("FILE")
                        .action(ArgAction::Append)
                )
        );
    let matches = app.clone().get_matches();

    // FIXME: don't read all sudokus into buffer
    //        read some in, process and output, read next in
    //let mut sudoku_buffer = String::new();

    if let Some(matches) = matches.subcommand_matches("solve") {
        let statistics = matches.contains_id("statistics");
        let action = actions_object(
            matches.contains_id("parallel"),
            matches.contains_id("no_parallel"),
        );

        // without printing solutions, print the header once
        // with solutions print it just before statistics
        if statistics {
            println!(
                "{:>9} {:>9} {:>9} {:>9} {:>10} {:>10} ",
                "total", "unique", "nonunique", "invalid", "time [s]", "sudokus/s"
            );
        }

        let action = |path: Option<&std::path::Path>, buffer: &str| match statistics {
            false => action.solve_and_print(buffer, path),
            true => action.solve_and_print_stats(buffer, path, statistics),
        };
        read_sudokus_and_execute(matches, action);
    } else if let Some(matches) = matches.subcommand_matches("generate") {

        let amount = *matches.get_one::<usize>("amount").unwrap();

        let gen_sud = match matches.is_present("solved") {
            true => Sudoku::generate_solved,
            false => Sudoku::generate,
        };
        let action = actions_object(
            matches.contains_id("parallel"),
            matches.contains_id("no_parallel"),
        );
        let print_blocks = matches.contains_id("print-block");

        action.gen_sudokus(amount, gen_sud, print_blocks);
    } else if let Some(matches) = matches.subcommand_matches("shuffle") {


        let amount=*matches.get_one::<usize>("count").unwrap_or(&1);

        let action = |_: Option<&std::path::Path>, buffer: &str| {
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            for sudoku in buffer.lines().map(Sudoku::from_str_line) {
                let mut sudoku = match sudoku {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = eprintln!("invalid sudoku: {}", e);
                        continue;
                    }
                };

                for _ in 0..amount {
                    sudoku.shuffle();
                    let _ = writeln!(lock, "{}", sudoku);
                }
            }
        };
        read_sudokus_and_execute(matches, action);
    } else if let Some(matches) = matches.subcommand_matches("canonicalize") {
        // TODO: factor out sudoku reading into a (better) function
        let action = |_: Option<&std::path::Path>, buffer: &str| {
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            for sudoku in buffer.lines().map(Sudoku::from_str_line) {
                let sudoku = match sudoku {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = eprintln!("invalid sudoku: {}", e);
                        continue;
                    }
                };

                if let Some((canonical, automorphisms)) = sudoku.canonicalized() {
                    let _ = writeln!(lock, "{} {}", canonical, automorphisms);
                } else {
                    let _ = writeln!(lock, "{} not valid or not solved", sudoku);
                }
            }
        };
        read_sudokus_and_execute(matches, action);
    } else {
        let _ = app.print_help();
    }
}
