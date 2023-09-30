use clap::{arg, Parser, Subcommand};
use std::path::PathBuf;
const AUTHOR: &str = "
Author : Fabrice Corraire <antidote1911@gmail.com>
Github : https://github.com/Antidote1911/";

pub fn get_styles() -> clap::builder::Styles {
    clap::builder::Styles::styled()
        .usage(
            anstyle::Style::new()
                .bold()
                .underline()
                .fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::Yellow))),
        )
        .header(
            anstyle::Style::new()
                .bold()
                .underline()
                .fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::Yellow))),
        )
        .literal(
            anstyle::Style::new().fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::Green))),
        )
        .invalid(
            anstyle::Style::new()
                .bold()
                .fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::Red))),
        )
        .error(
            anstyle::Style::new()
                .bold()
                .fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::Red))),
        )
        .valid(
            anstyle::Style::new()
                .bold()
                .underline()
                .fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::Green))),
        )
        .placeholder(
            anstyle::Style::new().fg_color(Some(anstyle::Color::Ansi(anstyle::AnsiColor::White))),
        )
}


#[derive(Debug, Parser)]
#[command(styles=get_styles())]
#[command(about, author=AUTHOR, version)]
#[command(
help_template = "{about-section}Version: {version} {author} \n\n {usage-heading} \n {usage} \n\n {all-args} \n {tab}"
)]
pub struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Solve Sudoku
    #[command(arg_required_else_help = false)]
    Solve {
        /// List of files containing Sudoku. Directories are silently ignored.
        #[arg(value_name = "FILE", value_delimiter = ',')]
        sudokus_file: Vec<PathBuf>,

        /// Show only statistics. Solutions are not printed
        #[clap(short)]
        statistics: bool,

        /// Enable parallel computing
        #[clap(short, conflicts_with = "no_parallel")]
        parallel: bool,

        /// No parallel computing
        #[clap(short, conflicts_with = "parallel")]
        no_parallel: bool,
    },
    /// Generate Sudoku
    Generate {
        /// Amount to generate
        #[clap(short, long, default_value = "5")]
        amount: Option<usize>,
        /// Display sudoku as block
        #[clap(short)]
        block: bool,
        /// Generate solved sudoku
        #[clap(short)]
        solved: bool,
        /// Enable parallel computing
        #[clap(short, conflicts_with = "no_parallel")]
        parallel: bool,
        /// No parallel computing
        #[clap(short, conflicts_with = "parallel")]
        no_parallel: bool,
    },
    /// Create different, but equivalent sudokus
    Shuffle {
        /// List of files containing Sudoku. Directories are silently ignored.
        #[arg(value_name = "FILE", value_delimiter = ',')]
        sudokus_file: Vec<PathBuf>,
        /// Amount
        #[clap(short, long, default_value = "1")]
        amount: Option<usize>,
    },
    /// Find the lexicographically minimal, equivalent sudoku
    Canonicalize {
        /// List of files containing Sudoku. Directories are silently ignored.
        #[arg(value_name = "FILE", value_delimiter = ',')]
        sudokus_file: Vec<PathBuf>,
    },
}
