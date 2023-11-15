use burn::{
    config::Config,
    nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Config)]
pub struct BareAttentionConfig {
    heads: usize,
    dimension: usize,
}
pub struct BareAttention<B: Backend> {
    inner_attention: MultiHeadAttention<B>,
}

impl<B: Backend> BareAttention<B> {
    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.inner_attention
            .forward(MhaInput::new(query, key, value))
            .context
    }
}

impl BareAttentionConfig {
    pub fn init<B: Backend>(&self) -> BareAttention<B> {
        let cfg = MultiHeadAttentionConfig::new(self.dimension, self.heads);

        BareAttention {
            inner_attention: cfg.init(),
        }
    }
}
