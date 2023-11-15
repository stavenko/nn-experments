use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, ops::TensorOps, Int, Tensor},
};

mod loader;
mod types;
pub use loader::SyntheticNutritionLablesLoader;

pub struct SyntheticNutritionLablesBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Batcher<NutritionLabelTestSample<B>, NutritionLabelsBatch<B>>
    for SyntheticNutritionLablesBatcher<B>
{
    fn batch(&self, items: Vec<NutritionLabelTestSample<B>>) -> NutritionLabelsBatch<B> {
        let label_tensors = items
            .iter()
            .map(|sample| sample.label.clone().unsqueeze())
            .collect();

        let image_tensors = items
            .into_iter()
            .map(|sample| sample.image.unsqueeze())
            .collect();
        let labels = Tensor::cat(label_tensors, 0).to_device(&self.device);
        let images = Tensor::cat(image_tensors, 0).to_device(&self.device);
        tracing::info!(
            "loaded batch of {:?} and  {:?}",
            images.shape(),
            labels.shape()
        );

        NutritionLabelsBatch { images, labels }
    }
}

impl<B: Backend> SyntheticNutritionLablesBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Debug, Clone)]
pub struct NutritionLabelsBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub labels: Tensor<B, 2>,
}

#[derive(Debug, Clone)]
pub struct NutritionLabelTestSample<B: Backend> {
    pub image: Tensor<B, 3>,
    pub label: Tensor<B, 1>,
}
