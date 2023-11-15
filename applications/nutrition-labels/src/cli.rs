use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
pub struct Cli {
    #[arg(long)]
    pub templates_dir: PathBuf,

    #[arg(long)]
    pub artifacts_dir: String,
}
