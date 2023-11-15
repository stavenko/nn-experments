use burn::{
    nn::{conv::Conv2d, Linear},
    tensor::backend::Backend,
};
use perceiver::Perceiver;

pub struct PerceiverNutritionLabels<B: Backend> {
    input: Conv2d<B>,
    perceiver: Perceiver<B>,
    output: Linear<B>,
}
