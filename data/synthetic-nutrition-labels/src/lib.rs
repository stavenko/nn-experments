use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Tensor},
};

mod loader;
mod types;

struct SyntheticNutritionLablesBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Batcher<NutritionLabelTestSample<B>, NutritionLabelsBatch<B>>
    for SyntheticNutritionLablesBatcher<B>
{
    fn batch(&self, items: Vec<NutritionLabelTestSample<B>>) -> NutritionLabelsBatch<B> {}
}

impl<B: Backend> SyntheticNutritionLablesBatcher<B> {
    fn new(device: B::Device) -> Self {
        Self { device }
    }
}

struct NutritionLabelsBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub labels: Tensor<B, 2>,
}

struct NutritionLabelTestSample<B: Backend> {
    pub image: Tensor<B, 3>,
    pub label: Tensor<B, 1>,
}
