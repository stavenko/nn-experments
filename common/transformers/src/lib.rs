// pub mod attention;
// pub mod block;
// pub mod embeddings;
// pub mod positional_encoding;
//

use burn::tensor::{backend::Backend, Tensor};

pub struct A<B: Backend, const D: usize> {
    t: Tensor<B, D>,
}

impl<B, const D: usize> A<B, D>
where
    B: Backend,
{
    fn g(&self) -> Tensor<B, D> {
        self.t.clone()
    }
}
