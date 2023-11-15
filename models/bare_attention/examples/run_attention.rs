use bare_attention::BareAttentionConfig;
use burn::{
    backend::{
        wgpu::{AutoGraphicsApi, WgpuDevice},
        WgpuBackend,
    },
    tensor::Tensor,
};

fn main() -> anyhow::Result<()> {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    //  type MyAutodiffBackend = ADBackendDecorator<MyBackend>;
    let att = BareAttentionConfig::new(2, 4);
    let model = att.init::<MyBackend>();

    let device = WgpuDevice::default();
    let q = Tensor::from_floats([0.0, 1.0, 0.0, 1.0])
        .reshape([1, 1, 4])
        .to_device(&device);
    let k = Tensor::ones([1, 8, 4]).to_device(&device);
    let v = Tensor::ones([1, 8, 4]).to_device(&device);

    println!("q: {}", q);
    println!("k: {}", k);
    println!("v: {}", v);

    let out = model.forward(q, k, v);

    println!("out: {}", out);

    Ok(())
}
