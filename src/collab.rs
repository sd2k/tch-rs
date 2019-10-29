use crate::{
    nn::{self, Embedding, Module, ModuleT, Path, SequentialT},
    tensor::index::IndexOp,
    Kind, Tensor,
};
use std::{borrow::Borrow, cell::RefCell, collections::HashMap, ops::Range};

#[derive(Debug)]
pub struct EmbeddingDotBias {
    user_weight: Embedding,
    item_weight: Embedding,
    user_bias: Embedding,
    item_bias: Embedding,
    user_lookup: RefCell<HashMap<i64, i64>>,
    item_lookup: RefCell<HashMap<i64, i64>>,
    y_range: Option<Range<f64>>,
}

pub fn embedding<'a, T: Borrow<Path<'a>>>(
    vs: T,
    num_embeddings: i64,
    embedding_dim: i64,
) -> nn::Embedding {
    nn::embedding(
        vs,
        num_embeddings,
        embedding_dim,
        nn::EmbeddingConfig {
            sparse: false,
            scale_grad_by_freq: false,
            ws_init: nn::Init::TruncatedNormal {
                mean: 0.0,
                stdev: 1.0,
            },
        },
    )
}

impl EmbeddingDotBias {
    pub fn new<'a, T: Borrow<Path<'a>>>(
        vs: T,
        y_range: Option<Range<f64>>,
        n_factors: i64,
        n_users: i64,
        n_items: i64,
    ) -> Self {
        let vs = vs.borrow();
        EmbeddingDotBias {
            user_weight: embedding(vs / "user_weight", n_users, n_factors),
            item_weight: embedding(vs / "item_weight", n_items, n_factors),
            user_bias: embedding(vs / "user_bias", n_users, 1),
            item_bias: embedding(vs / "item_bias", n_items, 1),
            user_lookup: Default::default(),
            item_lookup: Default::default(),
            y_range,
        }
    }
}

impl Module for EmbeddingDotBias {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let users = xs.i(0);
        let items = xs.i(1);
        let users_idx: Tensor = {
            let mut user_lookup = self.user_lookup.borrow_mut();
            let vec: Vec<i64> = users
                .iter::<i64>()
                .unwrap()
                .map(|item: i64| {
                    let len = user_lookup.len();
                    *user_lookup.entry(item).or_insert_with(|| (len) as i64)
                })
                .collect();
            vec.as_slice().into()
        };
        let items_idx: Tensor = {
            let mut item_lookup = self.item_lookup.borrow_mut();
            let vec: Vec<i64> = items
                .iter::<i64>()
                .unwrap()
                .map(|item: i64| {
                    let len = item_lookup.len();
                    *item_lookup.entry(item).or_insert_with(|| (len) as i64)
                })
                .collect();
            vec.as_slice().into()
        };
        let dot = self.user_weight.forward(&users_idx) * self.item_weight.forward(&items_idx);
        let res = dot.sum1(&[1], false, Kind::Double)
            + self.user_bias.forward(&users_idx).squeeze()
            + self.item_bias.forward(&items_idx).squeeze();
        match &self.y_range {
            Some(y_range) => res.sigmoid() * (y_range.end - y_range.start) + y_range.start,
            None => res,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EmbeddingSpec {
    num_embeddings: i64,
    embedding_dim: i64,
}

impl From<(i64, i64)> for EmbeddingSpec {
    fn from(other: (i64, i64)) -> EmbeddingSpec {
        EmbeddingSpec {
            num_embeddings: other.0,
            embedding_dim: other.1,
        }
    }
}

/// Create a sequence of (optional) Batch Norm -> Dropout -> Linear -> (optional) ReLU layers.
///
/// `in_dim` and `out_dim` refer to the dimensions of the Linear layer.
///
/// Taken from `fastai.layers.bn_drop_lin`.
fn batch_norm_dropout_linear<'a, T: Borrow<Path<'a>>>(
    vs: T,
    in_dim: i64,
    out_dim: i64,
    batch_norm: bool,
    dropout_prob: f64,
    relu: bool,
) -> Vec<Box<dyn ModuleT>> {
    let vs = vs.borrow();
    let mut layers: Vec<Box<dyn ModuleT>> = Vec::new();
    if batch_norm {
        layers.push(Box::new(nn::batch_norm1d(
            vs / "batch_norm",
            in_dim,
            Default::default(),
        )));
    }
    if dropout_prob > 0.0 {
        layers.push(Box::new(nn::func_t(move |xs, train| {
            xs.dropout(dropout_prob, train)
        })));
    }
    layers.push(Box::new(nn::linear(
        vs / "linear",
        in_dim,
        out_dim,
        Default::default(),
    )));
    if relu {
        layers.push(Box::new(nn::func(|xs| xs.relu())));
    }
    layers
}

#[derive(Debug)]
pub struct EmbeddingNN {
    layers: SequentialT,
    user_embeddings: Embedding,
    item_embeddings: Embedding,
    embedding_dropout: f64,
    y_range: Option<Range<f64>>,
    user_lookup: RefCell<HashMap<i64, i64>>,
    item_lookup: RefCell<HashMap<i64, i64>>,
}

impl EmbeddingNN {
    pub fn new<'a, T: Borrow<Path<'a>>>(
        vs: T,
        y_range: Option<Range<f64>>,
        n_user_embeddings: EmbeddingSpec,
        n_item_embeddings: EmbeddingSpec,
        layers: &[i64],
        embedding_dropout: f64,
        batch_norm_final: bool,
    ) -> EmbeddingNN {
        let vs = vs.borrow();

        // This is just the `fastai.tabular.models.TabularModel.__init__`
        // implementation, but with some options removed for now.
        let user_embeddings = embedding(
            vs / "user_embeddings",
            n_user_embeddings.num_embeddings,
            n_user_embeddings.embedding_dim,
        );
        let item_embeddings = embedding(
            vs / "item_embeddings",
            n_item_embeddings.num_embeddings,
            n_item_embeddings.embedding_dim,
        );
        let n_embeddings = n_user_embeddings.embedding_dim + n_item_embeddings.embedding_dim;

        let sizes = get_layer_sizes(n_embeddings, layers);

        let dropout_probs = vec![0.0; sizes.len()];

        // Insert a ReLU at all but the last layer.
        let mut relus = vec![true; sizes.len() - 1];
        relus.push(false);

        // Insert a batch norm at all but the first layer.
        let mut batch_norms = vec![true; sizes.len() - 1];
        batch_norms.insert(0, false);

        let mut layers = nn::seq_t();
        for i in 0..sizes.len() {
            let bn_d_l_layers = batch_norm_dropout_linear(
                vs / format!("layer{}", i),
                sizes[i].0,
                sizes[i].1,
                batch_norms[i],
                dropout_probs[i],
                relus[i],
            );
            layers = layers.append(bn_d_l_layers);
        }
        if batch_norm_final {
            layers = layers.add(nn::batch_norm1d(
                vs / "batch_norm_final",
                sizes.iter().last().unwrap().1,
                Default::default(),
            ));
        }
        EmbeddingNN {
            layers,
            user_embeddings,
            item_embeddings,
            y_range,
            embedding_dropout,
            item_lookup: Default::default(),
            user_lookup: Default::default(),
        }
    }
}

/// Get the (input, output) sizes for each of the layers in an EmbeddingNN.
///
/// The input size for each layer will be equal to the output of the previous
/// layer, while the final layer will have size 1 (since we're outputting a
/// single f64).
fn get_layer_sizes(n_embeddings: i64, linear_layer_sizes: &[i64]) -> Vec<(i64, i64)> {
    let mut sizes = vec![n_embeddings];
    sizes.extend_from_slice(linear_layer_sizes);
    sizes.push(1i64);
    sizes.windows(2).map(|s| (s[0], s[1])).collect()
}

impl ModuleT for EmbeddingNN {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let users = xs.i(0);
        let items = xs.i(1);
        let users_idx: Tensor = {
            let mut user_lookup = self.user_lookup.borrow_mut();
            let vec: Vec<i64> = users
                .iter::<i64>()
                .unwrap()
                .map(|item: i64| {
                    let len = user_lookup.len();
                    *user_lookup.entry(item).or_insert_with(|| (len + 1) as i64)
                })
                .collect();
            vec.as_slice().into()
        };
        let items_idx: Tensor = {
            let mut item_lookup = self.item_lookup.borrow_mut();
            let vec: Vec<i64> = items
                .iter::<i64>()
                .unwrap()
                .map(|item: i64| {
                    let len = item_lookup.len();
                    *item_lookup.entry(item).or_insert_with(|| (len + 1) as i64)
                })
                .collect();
            vec.as_slice().into()
        };
        let user_x = self.user_embeddings.forward(&users_idx);
        let item_x = self.item_embeddings.forward(&items_idx);
        let x = Tensor::cat(&[user_x, item_x], 1);
        let x = if self.embedding_dropout >= 0.0 {
            x.dropout(self.embedding_dropout, train)
        } else {
            x
        };
        let x = self.layers.forward_t(&x, train);
        match &self.y_range {
            Some(y_range) => x.sigmoid() * (y_range.end - y_range.start) + y_range.start,
            None => x,
        }
    }
}

pub fn collab_emb_dot_bias(
    p: &nn::Path,
    y_range: Option<Range<f64>>,
    n_factors: i64,
    n_users: i64,
    n_items: i64,
) -> impl ModuleT {
    EmbeddingDotBias::new(p, y_range, n_factors, n_users, n_items)
}

// TODO - we should probably use the builder pattern here.
pub fn collab_emb_nn(
    p: &nn::Path,
    y_range: Option<Range<f64>>,
    n_user_embeddings: EmbeddingSpec,
    n_item_embeddings: EmbeddingSpec,
    layers: &[i64],
    embedding_dropout: f64,
    batch_norm_final: bool,
) -> impl ModuleT {
    EmbeddingNN::new(
        p,
        y_range,
        n_user_embeddings,
        n_item_embeddings,
        layers,
        embedding_dropout,
        batch_norm_final,
    )
}
