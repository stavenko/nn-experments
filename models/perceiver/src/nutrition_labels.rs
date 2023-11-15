use burn::{
    tensor::backend::{ADBackend, Backend},
    train::{ClassificationOutput, RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use log::info;
use synthetic_nutrition_labels::NutritionLabelsBatch;

use crate::Perceiver;

impl<B: ADBackend> TrainStep<NutritionLabelsBatch<B>, RegressionOutput<B>> for Perceiver<B> {
    fn step(&self, batch: NutritionLabelsBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_classification(batch.images, batch.labels);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<NutritionLabelsBatch<B>, RegressionOutput<B>> for Perceiver<B> {
    fn step(&self, batch: NutritionLabelsBatch<B>) -> RegressionOutput<B> {
        self.forward_classification(batch.images, batch.labels)
    }
}
