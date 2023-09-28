use clap::{arg, Args, Parser, Subcommand, ValueEnum};
use std::ffi::OsString;
use std::path::PathBuf;

/// A Sudoku solver and generator

#[derive(Debug, Parser)]
#[command(about = "Ultra fast parallelized Sudoku solver, generator and more.", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands,

}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Solve Sudoku
    #[command(arg_required_else_help = false)]
    Solve {
        /// Solve Sudoku
        #[arg(value_name = "FILE",value_delimiter = ',')]
        sudokus_file: Option<String>,

        /// Show statistics
        #[clap(short)]
        statistics: bool,

        /// Enable parallel
        #[clap(short,conflicts_with = "no_parallel")]
        parallel: bool,

        /// No parallel
        #[clap(short,conflicts_with = "parallel")]
        no_parallel: bool,
    },
    /// Generate Sudoku
    Generate {
        #[clap(short, long, default_value = "5")]
        amount: Option<usize>,
        #[clap(short)]
        block: bool,
        #[clap(short)]
        solved: bool,
        #[clap(short,conflicts_with = "no_parallel")]
        parallel: bool,
        #[clap(short,conflicts_with = "parallel")]
        no_parallel: bool,

    },
    /// pushes things
    #[command(arg_required_else_help = true)]
    Push {
        /// The remote to target
        remote: String,
    },
    /// adds things
    #[command(arg_required_else_help = true)]
    Add {
        /// Stuff to add
        #[arg(required = true)]
        path: Vec<PathBuf>,
    },
    Stash(StashArgs),
    #[command(external_subcommand)]
    External(Vec<OsString>),
}

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
enum ColorWhen {
    Always,
    Auto,
    Never,
}

impl std::fmt::Display for ColorWhen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_possible_value()
            .expect("no values are skipped")
            .get_name()
            .fmt(f)
    }
}

#[derive(Debug, Args)]
#[command(args_conflicts_with_subcommands = true)]
struct StashArgs {
    #[command(subcommand)]
    command: Option<StashCommands>,

    #[command(flatten)]
    push: StashPushArgs,
}

#[derive(Debug, Subcommand)]
enum StashCommands {
    Push(StashPushArgs),
    Pop { stash: Option<String> },
    Apply { stash: Option<String> },
}

#[derive(Debug, Args)]
struct StashPushArgs {
    #[arg(short, long)]
    message: Option<String>,
}

