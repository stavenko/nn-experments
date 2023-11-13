use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, Distribution, Tensor},
};

#[derive(Config)]
pub struct PerceiverConfig {
    input_dim: [usize; 2],
    output_dim: [usize; 2],
    latents_dim: [usize; 2],
    attention_heads: usize,
    attention_dimension: usize,
    encoder_config: EncoderConfig,
    decoder_config: DecoderConfig,
    processor_config: ProcessorConfig,
}

#[derive(Config)]
pub struct DecoderConfig {
    attention_heads: usize,
    attention_dimension: usize,
    attention_layers: usize,
    output_dim: [usize; 2],
}

#[derive(Config)]
pub struct EncoderConfig {
    attention_layers: usize,
    attention_heads: usize,
    attention_dimension: usize,
}

#[derive(Config)]
pub struct ProcessorConfig {
    attention_layers: usize,
    attention_heads: usize,
    attention_dimension: usize,
}

#[derive(Config)]
pub struct AttentionBlockConfig {
    attention_heads: usize,
    attention_dimension: usize,
}

impl AttentionBlockConfig {
    fn init<B: Backend>(&self) -> AttentionBlock<B> {
        let mlp_input = self.attention_dimension * self.attention_heads;

        AttentionBlock {
            attention: MultiHeadAttentionConfig::new(
                self.attention_dimension,
                self.attention_heads,
            )
            .init(),
            multi_layer_perceptron: LinearConfig::new(mlp_input, mlp_input).init(),
            normalization: LayerNormConfig::new(mlp_input).init(),
        }
    }
}

impl ProcessorConfig {
    fn init<B: Backend>(&self) -> Processor<B> {
        let mut attentions = Vec::with_capacity(self.attention_layers);

        for _ in 0..self.attention_layers {
            let attention_block = AttentionBlockConfig {
                attention_heads: self.attention_heads,
                attention_dimension: self.attention_dimension,
            }
            .init();
            attentions.push(attention_block)
        }

        Processor { attentions }
    }
}

impl DecoderConfig {
    fn init<B: Backend>(&self) -> Decoder<B> {
        let attention = AttentionBlockConfig {
            attention_heads: self.attention_heads,
            attention_dimension: self.attention_dimension,
        }
        .init();
        let output_query = Tensor::random(
            [1, self.output_dim[0], self.output_dim[1]],
            Distribution::Default,
        )
        .into();

        Decoder {
            output_query,
            attention,
        }
    }
}

impl EncoderConfig {
    fn init<B: Backend>(&self) -> Encoder<B> {
        let mut attentions = Vec::with_capacity(self.attention_layers);

        let mlp_input = self.attention_dimension * self.attention_heads;
        for _ in 0..self.attention_layers {
            let attention_block = AttentionBlockConfig {
                attention_heads: self.attention_heads,
                attention_dimension: self.attention_dimension,
            }
            .init();
            attentions.push(attention_block)
        }

        Encoder { attentions }
    }
}

#[derive(Module, Debug)]
pub struct Perceiver<B: Backend> {
    latents: Param<Tensor<B, 3>>,
    encoder: Encoder<B>,
    processor: Processor<B>,
    decoder: Decoder<B>,
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    attentions: Vec<AttentionBlock<B>>,
}

#[derive(Module, Debug)]
pub struct AttentionBlock<B: Backend> {
    attention: MultiHeadAttention<B>,
    multi_layer_perceptron: burn::nn::Linear<B>,
    normalization: burn::nn::LayerNorm<B>,
}

impl<B: Backend> AttentionBlock<B> {
    fn forward(&self, query: Tensor<B, 3>, key: Tensor<B, 3>, value: Tensor<B, 3>) -> Tensor<B, 3> {
        let att = self.attention.forward(MhaInput::new(query, key, value));
        let normalized = self.normalization.forward(att.context);
        self.multi_layer_perceptron.forward(normalized)
    }
}

impl<B: Backend> Encoder<B> {
    fn forward(&self, latents: Tensor<B, 3>, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let key = input.clone();
        let value = input.clone();
        let mut latents = latents;
        for att in &self.attentions {
            latents = att.forward(latents, key.clone(), value.clone());
        }
        latents
    }
}

impl<B: Backend> Decoder<B> {
    fn forward(&self, latents: Tensor<B, 3>) -> Tensor<B, 3> {
        let query = self.output_query.val();
        let key = latents.clone();
        let value = latents.clone();
        let result = self.attention.forward(query, key, value);

        result
    }
}

impl<B: Backend> Processor<B> {
    fn forward(&self, latents: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut latents = latents;
        for att in &self.attentions {
            let query = latents.clone();
            let key = latents.clone();
            let value = latents.clone();
            latents = att.forward(query, key, value);
        }
        latents
    }
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    output_query: Param<Tensor<B, 3>>,
    attention: AttentionBlock<B>,
}

#[derive(Module, Debug)]
pub struct Processor<B: Backend> {
    attentions: Vec<AttentionBlock<B>>,
}

impl<B: Backend> Perceiver<B> {
    fn encode(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let key = input.clone();
        let value = input.clone();
        let latents = self.encoder.forward(self.latents.val(), input);
        latents
    }

    fn process(&self, latents: Tensor<B, 3>) -> Tensor<B, 3> {
        let latents = self.processor.forward(latents);
        latents
    }

    fn decode(&self, latents: Tensor<B, 3>) -> Tensor<B, 3> {
        self.decoder.forward(latents)
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let new_latents = self.encode(input);
        let new_latents = self.process(new_latents);
        let output = self.decode(new_latents);

        output
    }
}

impl PerceiverConfig {
    pub(crate) fn init<B: Backend>(self) -> Perceiver<B> {
        Perceiver {
            encoder: self.encoder_config.init(),
            decoder: self.decoder_config.init(),
            processor: self.processor_config.init(),
            latents: Tensor::random(
                [1, self.latents_dim[0], self.latents_dim[1]],
                Distribution::Default,
            )
            .into(),
        }
    }
}
