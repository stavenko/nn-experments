use burn::tensor::{backend::Backend, Int, Tensor, TensorKind};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct NutritionValue {
    calories: u16,
    proteins: u8,
    fats: u8,
    carbohydrates: u8,
}

impl NutritionValue {
    pub fn to_bytes(&self) -> [u8; 5] {
        let mut bytes = [0, 0, self.proteins, self.fats, self.carbohydrates];
        bytes[0..2].copy_from_slice(&self.calories.to_be_bytes());

        bytes
    }

    pub(crate) fn to_tensor<B: Backend>(&self, device: B::Device) -> Tensor<B, 1, Int> {
        let bytes = self.to_bytes().map(|b| b as i32);
        let tensor = Tensor::from_ints(bytes);
        tensor.to_device(&device)
    }

    pub fn create_random_value() -> Self {
        let calories = rand::random();
        let proteins = rand::random();
        let fats = rand::random();
        let carbohydrates = rand::random();

        let y = Self {
            calories,
            proteins,
            fats,
            carbohydrates,
        };

        y
    }
}
