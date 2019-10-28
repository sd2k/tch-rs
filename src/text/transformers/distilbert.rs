use std::f64::consts::FRAC_1_SQRT_2;

use crate::{
    nn,
    nn::{Module, ModuleT},
    Kind, Tensor,
};

fn gelu(t: &Tensor) -> Tensor {
    0.5 * t * &(1.0 + (t * FRAC_1_SQRT_2).erf())
}

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Gelu,
    Relu,
}

#[derive(Clone, Debug)]
struct Config {
    num_labels: i64,
    vocab_size: i64,
    max_position_embeddings: i64,
    sinusoidal_pos_embeddings: bool,
    n_layers: i64,
    n_heads: i64,
    dim: i64,
    hidden_dim: i64,
    dropout: f64,
    attention_dropout: f64,
    activation: Activation,
    initializer_range: f64,
    tie_weights_: bool,
    qa_dropout: f64,
    seq_classif_dropout: f64,
}

impl Config {
    fn hidden_size(&self) -> i64 {
        self.dim
    }

    fn num_attention_heads(&self) -> i64 {
        self.n_heads
    }

    fn num_hidden_layers(&self) -> i64 {
        self.n_layers
    }
}

impl Default for Config {
    fn default() -> Self {
        Config {
            num_labels: 2,
            vocab_size: 30522,
            max_position_embeddings: 512,
            sinusoidal_pos_embeddings: false,
            n_layers: 6,
            n_heads: 12,
            dim: 768,
            hidden_dim: 4 * 768,
            dropout: 0.1,
            attention_dropout: 0.1,
            activation: Activation::Gelu,
            initializer_range: 0.02,
            tie_weights_: true,
            qa_dropout: 0.1,
            seq_classif_dropout: 0.2,
        }
    }
}

#[derive(Debug)]
struct Embeddings {
    word_embeddings: nn::Embedding,
    position_embeddings: nn::Embedding,
    layer_norm: i64,
    dropout: f64,
}

impl Embeddings {
    fn new(vs: &nn::Path, config: &Config) -> Embeddings {
        Embeddings {
            word_embeddings: nn::embedding(
                vs / "word_embeddings",
                config.vocab_size,
                config.dim,
                Default::default(),
            ),
            position_embeddings: nn::embedding(
                vs / "position_embeddings",
                config.max_position_embeddings,
                config.dim,
                Default::default(),
            ),
            layer_norm: config.dim,
            dropout: config.dropout,
        }
    }
}

impl ModuleT for Embeddings {
    fn forward_t(&self, input_ids: &Tensor, train: bool) -> Tensor {
        let seq_length = input_ids.size1().unwrap();
        let position_ids = Tensor::arange(seq_length, (Kind::Int64, input_ids.device()))
            .unsqueeze(0)
            .expand_as(input_ids);
        let word_embeddings = self.word_embeddings.forward(input_ids);
        let position_embeddings = self.position_embeddings.forward(&position_ids);
        let embeddings = word_embeddings + position_embeddings;
        let embeddings = embeddings.layer_norm(&[self.layer_norm], None, None, 1e-12, true);
        let embeddings = embeddings.dropout(self.dropout, train);
        embeddings
    }
}

fn pre_classifier(p: nn::Path, dim: i64) -> impl ModuleT {
    nn::linear(p, dim, dim, Default::default())
}

fn classifier_(p: nn::Path, dim: i64, num_labels: i64) -> impl ModuleT {
    nn::linear(p, dim, num_labels, Default::default())
}

pub fn classifier(p: &nn::Path, config: Config) -> impl ModuleT {
    nn::seq_t()
        .add(distilbert(config))
        .add(pre_classifier(p / "pre_classifier", config.dim))
        .add(classifier_(p / "classifier", config.dim, config.num_labels))
        .add_fn_t(|xs, train| xs.dropout(config.seq_classif_dropout, train))
}

// pub struct DistilBert {}

// pub fn distil_bert() -> impl ModuleT {}
