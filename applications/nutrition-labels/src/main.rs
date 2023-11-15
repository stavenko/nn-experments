use burn::{
    autodiff::ADBackendDecorator,
    backend::{
        wgpu::{AutoGraphicsApi, WgpuDevice},
        WgpuBackend,
    },
};
use clap::Parser;
use perceiver::{get_config, train};

mod cli;
mod model;
mod train;

#[cfg(test)]
mod test;

fn main() -> anyhow::Result<()> {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = ADBackendDecorator<MyBackend>;

    let cmd = cli::Cli::parse();
    let config = get_config(&cmd.artifacts_dir)?;
    let device = WgpuDevice::default();
    train::<MyAutodiffBackend>(cmd.templates_dir, &cmd.artifacts_dir, config, device)?;

    Ok(())
}
