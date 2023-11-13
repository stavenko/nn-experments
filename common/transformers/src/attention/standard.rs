//! Various forms of multi-head attention described in paper "Attention Is All You Need".

use crate::block::TransformerBlockConfig;

use super::general::{
    AttentionStrategy, CausalMhaConfig, GeneralMultiHeadAttention, GeneralMultiHeadSelfAttention,
    MhaConfig, SelfMhaConfig,
};
use tch::{nn, Tensor};

const A4D_TENSOR: &str = "Tensor must be 4-dimensional";

/// Ensures that each position (of a decoder's self-attention) cannot attend
/// to subsequent positions. Such connections in a QK matrix are represented by items
/// above the diagonal. So we assign -inf (or some large negative number)
/// to all invalid connections, and later softmax will turn them into zeros.
///
/// We need this to guarantee that decoder's predictions are based
/// on what has happened before the position, not after.
/// Argument `qkt` is a scaled dot-product of Q and K.T,
/// shaped as `(batch, num_heads, q_seq_len, k_seq_len)`.
fn causal_attention_mask(qkt: Tensor) -> Tensor {
    // Practically, q_seq_len and k_seq_len will always be the same
    let (_, _, q_seq_len, k_seq_len) = qkt.size4().unwrap();
    // Creates a boolean mask filled with `false` on and below the diagonal.
    // TODO: Can be cached using q_seq_len, k_seq_len and the device as a key
    let causal_mask = Tensor::ones(
        &[1, 1, q_seq_len, k_seq_len],
        (burn::Kind::Bool, qkt.device()),
    )
    .triu(1);
    // Applies the mask, replacing all above the diagonal with -inf
    qkt.masked_fill(&causal_mask, f64::NEG_INFINITY)
}

/// Calculates the output of the attention after the affine transformations
/// of the inputs were done. Concatenates the outputs of all heads,
/// without projecting them with the output projection.
/// Expects arguments and shapes:
///
/// * `q`: (batch_size, q_seq_len, num_heads, d_k)
/// * `v`: (batch_size, v_seq_len, num_heads, d_v)
/// * `k`: (batch_size, k_seq_len, num_heads, d_k)
/// * `attention_mask`: 1s or 0s for keys that can or should not be attended,
///    shaped as (batch_size, k_seq_len)
/// * `causal`: whether the causal mask must be applied to the QK product.
///
/// Returns a tensor shaped as `(batch_size, q_seq_len, num_heads * d_v)`
fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attention_mask: Option<&Tensor>,
    causal: bool,
) -> Tensor {
    let (batch_size, q_seq_len, num_heads, _) = q.size4().expect(A4D_TENSOR);
    let (_, _, _, key_value_dim) = v.size4().expect(A4D_TENSOR);
    let sqrt_d_k = (key_value_dim as f64).sqrt();
    // Q * transposed(K) / sqrt(d_k)
    // The result will have shape of (batch, num_heads, q_seq_len, k_seq_len)
    let mut qkt_scaled = q.transpose(2, 1).matmul(
        // k is transposed into (batch, num_heads, d_k, seq_len)
        &k.permute(&[0, 2, 3, 1]),
    ) / sqrt_d_k;
    if let Some(attention_mask) = attention_mask {
        // setting keys we do not need to attend to a large negative, so that softmax
        // will turn them into zeroes
        qkt_scaled += -1e5 * (1.0 - attention_mask.view([batch_size, 1, 1, -1]));
    }
    if causal {
        qkt_scaled = causal_attention_mask(qkt_scaled)
    }
    let scaled_attention = qkt_scaled
        .softmax(-1, qkt_scaled.kind())
        .matmul(&v.transpose(2, 1));
    // "Concatenating" heads by rearranging dimensions
    // from (batch, num_heads, seq_len, key_value_dim)
    // and reshaping the result
    scaled_attention
        .transpose(2, 1)
        .reshape(&[-1, q_seq_len, num_heads * key_value_dim])
}

#[derive(Debug, Clone)]
pub struct StandardMhaConfig {
    pub num_heads: usize,
    pub head_key_query_dim: usize,
    pub head_value_dim: usize,
    pub input_query_dim: usize,
    pub input_key_value_dim: usize,
    pub output_dim: usize,
    pub causal: bool,
}

#[rustfmt::skip]
impl MhaConfig for StandardMhaConfig {
    fn num_heads(&self) -> usize { self.num_heads }
    fn head_key_query_dim(&self) -> usize { self.head_key_query_dim }
    fn head_value_dim(&self) -> usize { self.head_value_dim }
    fn input_query_dim(&self) -> usize { self.input_query_dim }
    fn input_key_value_dim(&self) -> usize { self.input_key_value_dim }
    fn output_dim(&self) -> usize { self.output_dim }
}

impl CausalMhaConfig for StandardMhaConfig {
    fn is_causal(&self) -> bool {
        self.causal
    }
}

/// Canonical multi-head attention described in paper
/// ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf).
///
/// It accepts 2 separate inputs: one for queries and another for the keys/values.
/// Input for keys and values can have sequence length different from the sequence
/// length of queries.
///
/// This allows for construction of encoder-decoder models, in which a decoder
/// "interrogates" through its queries whatever values were produced by an encoder.
#[derive(Debug)]
pub struct StandardMultiHeadAttentionStrategy<ConfigType> {
    config: ConfigType,
}

impl<ConfigType> AttentionStrategy for StandardMultiHeadAttentionStrategy<ConfigType>
where
    ConfigType: CausalMhaConfig + std::fmt::Debug + Clone,
{
    type Config = ConfigType;

    fn new(_vs: nn::Path, config: &Self::Config) -> Self {
        Self {
            config: config.clone(),
        }
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        _training: bool,
    ) -> Tensor {
        attention(q, k, v, attention_mask, self.config.is_causal())
    }
}

#[derive(Clone, Debug)]
pub struct StandardSelfMhaConfig {
    /// Last dimension of input from which queries, keys and values are constructed
    pub input_dim: usize,
    /// number of attention heads
    pub num_heads: usize,
    pub head_key_query_value_dim: usize,
    pub output_dim: usize,
    pub causal: bool,
}

#[rustfmt::skip]
impl SelfMhaConfig for StandardSelfMhaConfig {
    fn input_dim(&self) -> usize { self.input_dim }
    fn num_heads(&self) -> usize { self.num_heads }
    fn head_key_query_value_dim(&self) -> usize { self.head_key_query_value_dim }
    fn output_dim(&self) -> usize { self.output_dim }
}

impl CausalMhaConfig for StandardSelfMhaConfig {
    fn is_causal(&self) -> bool {
        self.causal
    }
}

pub type StandardMultiHeadAttention =
    GeneralMultiHeadAttention<StandardMultiHeadAttentionStrategy<StandardMhaConfig>>;
pub type StandardMultiHeadSelfAttention =
    GeneralMultiHeadSelfAttention<StandardMultiHeadAttentionStrategy<StandardSelfMhaConfig>>;

impl From<TransformerBlockConfig> for StandardSelfMhaConfig {
    fn from(value: TransformerBlockConfig) -> Self {
        Self {
            input_dim: value.input_output_dim,
            num_heads: value.num_heads,
            head_key_query_value_dim: value.key_query_value_dim,
            output_dim: value.input_output_dim,
            causal: value.causal,
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_attention() {
        let batch_size = 1;
        let d_k = 5;
        let num_heads = 3;
        let seq_len = 2;
        let qkv_dim = [batch_size, seq_len, num_heads, d_k];

        let key = tch::Tensor::of_slice(&[
            -0.5917, 0.1766, 0.4846, 0.3123, 0.2224, 0.1224, 0.2628, -1.4355, -0.2553, 0.3784,
            1.6483, -0.2301, -0.7826, 1.9004, -0.6406, 0.2766, 0.9230, 0.2658, -0.1112, -0.8449,
            -0.5094, 0.9149, 1.2510, 0.7405, 2.0509, 0.1292, -0.1290, -0.5136, -0.3277, -1.2158,
        ])
        .reshape(&qkv_dim);
        let query = tch::Tensor::of_slice(&[
            -1.4838, 0.1216, -0.6428, 1.0730, 0.0514, 1.0894, -1.2976, 0.3055, 0.4674, -0.7904,
            0.5942, 1.0004, -1.0341, 0.8607, -0.4123, 0.4504, -1.3332, 0.0440, -0.8076, 1.1087,
            0.3849, 0.1982, -0.9366, -0.3024, 1.7482, -0.5707, 0.1702, 1.5397, 1.0245, -0.2351,
        ])
        .reshape(&qkv_dim);
        let value = tch::Tensor::of_slice(&[
            -1.7960, 0.3486, -1.2651, -0.2911, -0.3844, -0.5393, -0.9316, -0.1701, -1.0799,
            -0.5474, 0.6255, 0.2315, -1.1439, 1.2168, -0.0378, -0.3315, -0.7856, 1.9734, -0.6110,
            0.8073, 1.2904, -1.1004, 0.4783, -0.8432, 0.5684, -1.0430, -0.1173, -0.2158, 2.0082,
            -1.7302,
        ])
        .reshape(&qkv_dim);

        #[rustfmt::skip]
        let expect_result = tch::Tensor::of_slice(&[
            -1.31020674, -0.02762855, -0.1908484, -0.39721489, 0.01090203,
            0.0669134, -0.98752656, 0.0447269, -1.00147692, -0.17771485,
            0.24940618, 0.15287757, -0.93469852, 1.39518816, -0.41928108,
            -1.2939129, -0.04024752, -0.15481726, -0.40077406, 0.02416073,
            0.37404134, -1.01586082, 0.15356537, -0.96174517, 0.00957998,
            -0.05222779, 0.08982097, -0.76691518, 1.53825866, -0.72523573,
        ])
        .reshape(&[batch_size, seq_len, num_heads * d_k]);

        #[rustfmt::skip]
        let expect_causal_result = tch::Tensor::of_slice(&[
           -1.796, 0.3486, -1.2651,
            -0.2911, -0.3844, -0.5393,
            -0.9316, -0.1701, -1.0799,
            -0.5474, 0.6255, 0.2315,
            -1.1439, 1.2168, -0.0378,
            -1.2939129019362632, -0.04024751561890764, -0.15481726385837366,
            -0.4007740612294909, 0.024160733876787365, 0.3740413384758751,
            -1.015860817584701, 0.15356536802085446, -0.9617451687067607,
            0.009579977849582743, -0.05222779442962869, 0.08982097411024606,
            -0.766915183691856, 1.5382586613794478, -0.725235732270125
        ]).reshape(&[batch_size, seq_len, num_heads * d_k]);

        let result = super::attention(&query, &key, &value, None, false);
        let diff = (&expect_result - &result)
            .square()
            .sum(expect_result.kind());
        assert!(f64::from(&diff) < 1e-9);

        let causal_result = super::attention(&query, &key, &value, None, true);
        let diff = (&expect_causal_result - &causal_result)
            .square()
            .sum(expect_result.kind());
        assert!(f64::from(&diff) < 1e-9);
    }
}
