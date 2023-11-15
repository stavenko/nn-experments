use burn::{nn::conv::Conv2d, tensor::backend::Backend};
use perceiver::Perceiver;

struct NutritionViewerConfig {}

struct NutritionViewer<B: Backend> {
    conv: Conv2d<B>,
    perceiver: Perceiver<B>,
}
