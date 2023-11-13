use burn::{
    tensor::backend::{ADBackend, Backend},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use synthetic_nutrition_labels::NutritionLabelsBatch;

use crate::Perceiver;

impl<B: ADBackend> TrainStep<NutritionLabelsBatch<B>, ClassificationOutput<B>> for Perceiver<B> {
    fn step(&self, batch: NutritionLabelsBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<NutritionLabelsBatch<B>, ClassificationOutput<B>> for Perceiver<B> {
    fn step(&self, batch: NutritionLabelsBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.labels)
    }
}
