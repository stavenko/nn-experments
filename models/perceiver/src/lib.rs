use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        loss::{MSELoss, Reduction},
        LayerNormConfig, LinearConfig,
    },
    tensor::{backend::Backend, Distribution, Tensor},
    train::RegressionOutput,
};
mod nutrition_labels;
mod training;
use ::log::info;
pub use training::*;

#[derive(Config)]
pub struct PerceiverConfig {
    latents_dim: [usize; 2],
    output_dim: [usize; 2],
    encoder: EncoderConfig,
    decoder: DecoderConfig,
    processor: ProcessorConfig,
}

#[derive(Config)]
pub struct DecoderConfig {
    attention: AttentionBlockConfig,
    output_dim: [usize; 2], // Output block dimensions: one for something, one for amount of values,
}

#[derive(Config)]
pub struct EncoderConfig {
    attention: AttentionBlockConfig,
}

#[derive(Config)]
pub struct ProcessorConfig {
    #[config(default = 8)]
    attention_layers: usize,
    attention: AttentionBlockConfig,
}

#[derive(Config)]
pub struct AttentionBlockConfig {
    #[config(default = 8)]
    heads: usize,
    dimensions: [usize; 2],
    perceptron_input: usize,
    // #[config(default = "None")]
    // perceptron_output: Option<usize>,
}

/// This attention block config is made with great respect to current task - maintain correct
/// initialization of this block in different parts of this model
/// This block has same traits, so I bother to reuse them.
impl AttentionBlockConfig {
    fn init<B: Backend>(&self) -> AttentionBlock<B> {
        let total_features = self.dimensions[0] * self.dimensions[1];

        AttentionBlock {
            mlp_dim: self.perceptron_input as i32,
            attention: MultiHeadAttentionConfig::new(self.dimensions[1], self.heads).init(),
            multi_layer_perceptron: LinearConfig::new(self.perceptron_input, self.perceptron_input)
                .init(),
            normalization: LayerNormConfig::new(total_features).init(),
        }
    }
}

impl ProcessorConfig {
    fn init<B: Backend>(&self) -> Processor<B> {
        let mut attentions = Vec::with_capacity(self.attention_layers);

        for _ in 0..self.attention_layers {
            let attention_block = self.attention.init();
            attentions.push(attention_block)
        }

        Processor { attentions }
    }
}

impl DecoderConfig {
    pub fn create(output_dim: [usize; 2], latents_dim: [usize; 2]) -> Self {
        let attention_block_config = AttentionBlockConfig {
            heads: 1,
            dimensions: output_dim,
            perceptron_input: output_dim[1],
        };
        Self {
            attention: attention_block_config,
            output_dim,
        }
    }
    fn init<B: Backend>(&self) -> Decoder<B> {
        let attention = self.attention.init();
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
        Encoder {
            attention: self.attention.init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Perceiver<B: Backend> {
    latents: Param<Tensor<B, 2>>,
    output_query: Param<Tensor<B, 2>>,
    encoder: Encoder<B>,
    processor: Processor<B>,
    decoder: Decoder<B>,
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    attention: AttentionBlock<B>,
}

#[derive(Module, Debug)]
pub struct AttentionBlock<B: Backend> {
    attention: MultiHeadAttention<B>,
    multi_layer_perceptron: burn::nn::Linear<B>,
    normalization: burn::nn::LayerNorm<B>,
    mlp_dim: i32,
}

impl<B: Backend> AttentionBlock<B> {
    fn forward(&self, query: Tensor<B, 3>, key: Tensor<B, 3>, value: Tensor<B, 3>) -> Tensor<B, 3> {
        // info!("lets apply attention of q");
        let att = self.attention.forward(MhaInput::new(query, key, value));
        // info!("result calculated {:?}", att.context.dims(),);
        let reshaped = att.context.reshape([0, 1, -1]);
        let normalized = self.normalization.forward(reshaped);
        // info!("normalized ");
        let reshaped = normalized.reshape([0, -1, self.mlp_dim]);
        let processed = self.multi_layer_perceptron.forward(reshaped);
        // info!("and processed from MLP");
        processed
    }
}

impl<B: Backend> Encoder<B> {
    fn forward(&self, latents: Tensor<B, 3>, input: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch_size, lw, lh] = latents.dims();
        let key = input.clone().reshape([0, -1, lh as i32]);
        let value = key.clone();
        let latents = latents;
        /*
        info!(
            "we took Q: {:?}, K: {:?} , V: {:?} ",
            [batch_size, lw, lh],
            key.shape(),
            value.shape()
        );
        */

        let latents = self.attention.forward(latents, key.clone(), value.clone());
        /*
        info!(
            "new latents successfully calculated, {:?}, lets reshape em",
            latents.shape()
        );
        */

        latents.reshape([-1, lw as i32, lh as i32])
    }
}

impl<B: Backend> Decoder<B> {
    fn forward(&self, latents: Tensor<B, 3>, output_query: Tensor<B, 3>) -> Tensor<B, 3> {
        let batch_size = latents.dims()[0];
        let key = latents
            .clone()
            .reshape([batch_size as i32, -1, output_query.dims()[2] as i32]);
        let value = key.clone();
        info!(
            "decoding: Q: {:?}, KV: {:?}",
            output_query.dims(),
            key.dims()
        );

        let result = self.attention.forward(output_query, key, value);

        result
    }
}

impl<B: Backend> Processor<B> {
    fn forward(&self, latents: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut latents = latents;
        let [batch_size, lw, lh] = latents.dims();
        for (layer, att) in self.attentions.iter().enumerate() {
            /*
            info!("---- layer {layer} -----");
            info!("Processing latents {:?}", latents.dims());
            */

            let query = latents.clone();
            let key = latents.clone();
            let value = latents;
            latents = att.forward(query, key, value).reshape([batch_size, lw, lh]);
            /*
            info!("got dims: {:?}", latents.dims());
            info!("========================");
            */
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
    fn encode(&self, input: Tensor<B, 4>, latents: Tensor<B, 3>) -> Tensor<B, 3> {
        let latents = self.encoder.forward(latents, input);
        latents
    }

    fn process(&self, latents: Tensor<B, 3>) -> Tensor<B, 3> {
        let latents = self.processor.forward(latents);
        latents
    }

    fn decode(&self, latents: Tensor<B, 3>, output_query: Tensor<B, 3>) -> Tensor<B, 3> {
        self.decoder.forward(latents, output_query)
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, _, _, _] = input.dims();
        let latents = self.latents.val().unsqueeze();
        let latents_tensors = (0..batch_size)
            .into_iter()
            .map(|_| latents.clone())
            .collect();

        let latents_batched = Tensor::cat(latents_tensors, 0);

        let output_query = self.output_query.val().unsqueeze();
        let output_query_tensors = (0..input.dims()[0])
            .into_iter()
            .map(|_| output_query.clone())
            .collect();

        let output_query_batched = Tensor::cat(output_query_tensors, 0);

        let encoded_latents = self.encode(input, latents_batched);
        let processed_latents = self.process(encoded_latents);
        let output = self.decode(processed_latents, output_query_batched);

        output.reshape([batch_size as i32, -1])
    }

    pub fn forward_classification(
        &self,
        input: Tensor<B, 4>,
        target: Tensor<B, 2>,
    ) -> RegressionOutput<B> {
        let output = self.forward(input);

        info!("output: {}", output);
        info!("expected: {}", target);

        let loss = MSELoss::new().forward(output.clone(), target.clone(), Reduction::Mean);
        info!("loss: {}", loss);
        RegressionOutput::new(loss, output, target)
    }
}

impl PerceiverConfig {
    pub(crate) fn init<B: Backend>(self) -> Perceiver<B> {
        Perceiver {
            encoder: self.encoder.init(),
            decoder: self.decoder.init(),
            processor: self.processor.init(),
            output_query: Tensor::random(self.output_dim, Distribution::Default).into(),
            latents: Tensor::random(
                [self.latents_dim[0], self.latents_dim[1]],
                Distribution::Default,
            )
            .into(),
        }
    }
}
