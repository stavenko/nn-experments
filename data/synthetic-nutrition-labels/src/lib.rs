use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, ops::TensorOps, Tensor},
};

mod loader;
mod types;

struct SyntheticNutritionLablesBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Batcher<NutritionLabelTestSample<B>, NutritionLabelsBatch<B>>
    for SyntheticNutritionLablesBatcher<B>
{
    fn batch(&self, items: Vec<NutritionLabelTestSample<B>>) -> NutritionLabelsBatch<B> {
        let labels = Tensor::cat(items.iter().map(|sample| sample.label.clone()).collect(), 0)
            .to_device(&self.device);
        let images = Tensor::cat(items.into_iter().map(|sample| sample.image).collect(), 0)
            .to_device(&self.device);
        NutritionLabelsBatch { images, labels }
    }
}

impl<B: Backend> SyntheticNutritionLablesBatcher<B> {
    fn new(device: B::Device) -> Self {
        Self { device }
    }
}

pub struct NutritionLabelsBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub labels: Tensor<B, 1>,
}

struct NutritionLabelTestSample<B: Backend> {
    pub image: Tensor<B, 3>,
    pub label: Tensor<B, 1>,
}
