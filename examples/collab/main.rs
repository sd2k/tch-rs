extern crate csv;
#[macro_use]
extern crate failure;
extern crate tch;

use std::path::Path;

use tch::{
    collab::{EmbeddingDotBias, EmbeddingNN},
    nn::{self, Module, ModuleT, OptimizerConfig},
    Kind, Tensor,
};

fn load_ratings<P: AsRef<Path>>(path: P) -> failure::Fallible<(Vec<i64>, Vec<i64>, Vec<f64>)> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut users = Vec::new();
    let mut items = Vec::new();
    let mut ratings = Vec::new();
    for result in reader.records() {
        let record = result?;
        users.push(record.get(0).unwrap().parse::<i64>().unwrap());
        items.push(record.get(1).unwrap().parse::<i64>().unwrap());
        ratings.push(record.get(2).unwrap().parse::<f64>().unwrap());
    }
    Ok((users, items, ratings))
}

fn mse_loss(x: &Tensor, y: &Tensor) -> Tensor {
    let diff = x - y;
    (&diff * &diff).mean(Kind::Float)
}

fn main() -> failure::Fallible<()> {
    let args: Vec<_> = std::env::args().collect();
    let ratings_path = match args.as_slice() {
        [_, p] => p.to_owned(),
        _ => bail!("usage: main ratings-csv"),
    };
    let (users_s, items_s, ratings_s) = load_ratings(ratings_path)?;
    let n_users = users_s
        .iter()
        .copied()
        .collect::<std::collections::HashSet<i64>>()
        .len() as i64;
    let n_items = items_s
        .iter()
        .copied()
        .collect::<std::collections::HashSet<i64>>()
        .len() as i64;
    let users = Tensor::of_slice(&users_s);
    let items = Tensor::of_slice(&items_s);
    let ratings = Tensor::of_slice(&ratings_s);
    let x = Tensor::cat(&[users, items], 0).reshape(&[2, -1]);

    let vs1 = tch::nn::VarStore::new(tch::Device::cuda_if_available());
    let model1 = EmbeddingDotBias::new(&vs1.root(), Some(0.0..5.5), 50, n_users, n_items);
    let mut opt1 = nn::Adam::default().build(&vs1, 5e-2)?;
    for epoch in 1..100 {
        let loss = mse_loss(&model1.forward(&x), &ratings);
        opt1.backward_step(&loss);
        println!("epoch: {:4} train loss: {:8.5}", epoch, f64::from(&loss));
    }
    println!(
        "User {:?}, item {:?}, actual {:?}, prediction {:?}",
        users_s[0],
        items_s[0],
        ratings_s[0],
        model1.forward(&Tensor::of_slice(&[users_s[0], items_s[0]]).reshape(&[2, 1])),
    );
    println!(
        "User {:?}, item {:?}, actual {:?}, prediction {:?}",
        users_s[1],
        items_s[1],
        ratings_s[1],
        model1.forward(&Tensor::of_slice(&[users_s[1], items_s[1]]).reshape(&[2, 1])),
    );

    let vs2 = tch::nn::VarStore::new(tch::Device::cuda_if_available());
    let model2 = EmbeddingNN::new(
        &vs2.root(),
        Some(0.0..5.5),
        (101, 21).into(),
        (101, 21).into(),
        &[50, 10],
        0.0,
        false,
    );
    let mut opt2 = nn::Adam::default().build(&vs2, 1e-2)?;
    for epoch in 1..200 {
        let loss = mse_loss(&model2.forward_t(&x, true), &ratings);
        opt2.backward_step(&loss);
        println!("epoch: {:4} train loss: {:8.5}", epoch, f64::from(&loss));
    }
    println!(
        "User {:?}, item {:?}, actual {:?}, prediction {:?}",
        users_s[0],
        items_s[0],
        ratings_s[0],
        model2
            .forward_t(
                &Tensor::of_slice(&[users_s[0], items_s[0]]).reshape(&[2, 1]),
                false
            )
            .squeeze(),
    );
    println!(
        "User {:?}, item {:?}, actual {:?}, prediction {:?}",
        users_s[1],
        items_s[1],
        ratings_s[1],
        model2
            .forward_t(
                &Tensor::of_slice(&[users_s[1], items_s[1]]).reshape(&[2, 1]),
                false
            )
            .squeeze(),
    );

    Ok(())
}
