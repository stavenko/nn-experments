use std::path::PathBuf;

use burn::data::dataset::Dataset;
use burn::train::logger;
use burn::{
    autodiff::ADBackendDecorator,
    backend::{
        self,
        wgpu::{AutoGraphicsApi, WgpuDevice},
        WgpuBackend,
    },
};
use synthetic_nutrition_labels::{SyntheticNutritionLablesBatcher, SyntheticNutritionLablesLoader};

#[test]
fn image_is_good() {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = ADBackendDecorator<MyBackend>;

    let device = WgpuDevice::default();

    let batcher   = SyntheticNutritionLablesLoader::<MyAutodiffBackend>::new(PathBuf::from("/Users/vasilijstavenko/projects/nn-experments/data/synthetic-nutrition-labels/data/templates/"),device.clone()).unwrap();

    let image = batcher.get(0).unwrap();

    assert!(false)
}
