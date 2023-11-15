use burn::tensor::{backend::Backend, Int, Tensor};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct NutritionValue {
    pub calories: f32,
    pub proteins: f32,
    pub fats: f32,
    pub carbohydrates: f32,
}

impl NutritionValue {
    pub fn create_random_value() -> Self {
        let mut rng = rand::thread_rng();
        let calories = rng.gen_range(0..1000) as f32;
        let proteins = rng.gen_range(0..100) as f32;
        let fats = rng.gen_range(0..100) as f32;
        let carbohydrates = rng.gen_range(0..100) as f32;

        let y = Self {
            calories,
            proteins,
            fats,
            carbohydrates,
        };

        y
    }

    pub(crate) fn as_array(&self) -> [f32; 4] {
        [self.calories, self.proteins, self.fats, self.carbohydrates]
    }
}
