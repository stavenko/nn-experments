use burn::nn::{self, LayerNormConfig, LinearConfig, GELU};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Allows to reuse the same word embedding matrix both for the input and
/// the output layers of the network.
/// This is called Weight Tying and is proven to improve performance
/// of neural network language models, as well as decrease their number
/// of parameters (eliminating the need for a separate huge matrix
/// of output weights).

/// The module is supposed to be called with a shared `embedded_matrix`, and
/// an input which comes from previous layer (like LSTM or Transformer).
/// `embedding_matrix` is supposed to be the shape of `[vocabulary_size, embedding_dim]`.
///
/// Based on papers:
///
/// * [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) by Press et al.
/// * [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling](https://arxiv.org/abs/1611.01462)
/// * [Improving language understanding with unsupervised learning](https://openai.com/research/language-unsupervised)
#[derive(Debug)]
pub struct TiedOutputEmbedding<B: Backend> {
    input_projection: nn::Linear<B>,
    layer_norm: nn::LayerNorm<B>,
    output_bias: Tensor<B, 1>,
    gelu: GELU,
}

impl<B: Backend> TiedOutputEmbedding<B> {
    pub fn new(input_dim: usize, embedding_dim: usize, vocab_size: usize) -> Self {
        // let projection = vs.kaiming_uniform("kernel", &[input_dim as i64, embedding_dim as i64]);
        let input_projection = LinearConfig::new(input_dim, embedding_dim).init();
        let layer_norm = LayerNormConfig::new(input_dim).init();
        let output_bias = Tensor::new(vocab_size as i64);
        Self {
            input_projection,
            layer_norm,
            output_bias,
            gelu: GELU::new(),
        }
    }

    pub fn forward(&self, input_embedding_matrix: &Tensor<B, 2>, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, input_dim] = xs.dims();
        let projected = self.input_projection.forward(xs);
        let projected = self.gelu.forward(projected);
        let projected = self.layer_norm.forward(projected);
        let projected = projected.reshape([usize::MAX, input_dim]);
        // matching with the embedding
        // the embedding matrix is expected to be the shape of (vocabulary_size, embedding_dim)
        // output shaped as (batch_size, seq_len, vocab_size)
        (projected.matmul(input_embedding_matrix.transpose()) + self.output_bias.reshape([1, -1]))
            .reshape(&[batch_size, seq_len, usize::MAX])
    }
}
