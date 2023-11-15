use std::path::PathBuf;

use burn::module::Module;
use burn::record::CompactRecorder;
use burn::train::metric::LossMetric;
use burn::train::LearnerBuilder;
use burn::{
    config::Config, data::dataloader::DataLoaderBuilder, optim::AdamConfig,
    tensor::backend::ADBackend,
};
use log::info;
use synthetic_nutrition_labels::{SyntheticNutritionLablesBatcher, SyntheticNutritionLablesLoader};

use crate::AttentionBlockConfig;
use crate::PerceiverConfig;

#[derive(Config)]
pub struct TrainingConfig {
    pub model: PerceiverConfig,
    pub optimizer: AdamConfig,
    #[config(default = 2)]
    pub num_epochs: usize,
    #[config(default = 10)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

pub fn get_config(artifact_dir: &str) -> anyhow::Result<TrainingConfig> {
    let _config_path = format!("{artifact_dir}/config.json");
    let latents_dim = [512, 1024];
    let output_dim = [1, 4];
    let processor_attention_block = AttentionBlockConfig::new(latents_dim, 256);
    let encoder_attention_block = AttentionBlockConfig::new(latents_dim, 256);
    let perceiver_config = PerceiverConfig::new(
        latents_dim,
        output_dim,
        crate::EncoderConfig::new(encoder_attention_block),
        crate::DecoderConfig::create(output_dim, latents_dim),
        crate::ProcessorConfig::new(processor_attention_block),
    );

    Ok(TrainingConfig::new(perceiver_config, AdamConfig::new()))
}

pub fn train<B: ADBackend>(
    synthetic_templates_dir: PathBuf,
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(artifact_dir)?;
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Save without error");

    B::seed(config.seed);

    let batcher_train = SyntheticNutritionLablesBatcher::<B>::new(device.clone());
    let batcher_valid = SyntheticNutritionLablesBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(SyntheticNutritionLablesLoader::new(
            synthetic_templates_dir.clone(),
            device.clone(),
        )?);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(SyntheticNutritionLablesLoader::new(
            synthetic_templates_dir.clone(),
            device.clone(),
        )?);

    let learner = LearnerBuilder::new(artifact_dir)
        // .metric_train_numeric(AccuracyMetric::new())
        // .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<B>(),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);
    info!("--- training complete ---");

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save trained model");

    Ok(())
}
