use burn::tensor::{Data, Shape};
use resvg::tiny_skia;
use resvg::usvg::{self, fontdb};
use resvg::usvg::{TreeParsing, TreeTextToPath};
use std::{
    fs,
    path::{Path, PathBuf},
};

use burn::tensor::{backend::Backend, Tensor};
use rand::seq::SliceRandom;

use crate::types::NutritionValue;

pub struct SyntheticNutritionLablesLoader {
    root_dir: PathBuf,
    template_filenames: Vec<String>,
    tera: tera::Tera,
    samples_available: usize,
}

impl SyntheticNutritionLablesLoader {
    pub fn new(templates_dir: PathBuf) -> anyhow::Result<Self> {
        let tera_glob = format!(
            "{}/*",
            templates_dir.to_str().ok_or(anyhow::Error::msg("wtf"))?
        );
        let mut tera = tera::Tera::new(&tera_glob)?;

        let templates_to_load = read_templates(&templates_dir)?;
        let samples_available = templates_to_load.len() * 100;
        let mut template_filenames = Vec::new();

        for file in templates_to_load {
            let file_name =
                file.file_name()
                    .map(|f| f.to_str())
                    .flatten()
                    .ok_or(anyhow::Error::msg(
                        "got some problems, on converting data to str",
                    ))?;

            tera.add_template_file(file.to_owned(), Some(file_name));
            template_filenames.push(file_name.to_string());
        }

        Ok(Self {
            root_dir: templates_dir.to_path_buf(),
            template_filenames,
            tera,
            samples_available,
        })
    }

    pub fn get_random_sample<B: Backend>(
        &mut self,
        device: B::Device,
    ) -> anyhow::Result<(Tensor<B, 3>, Tensor<B, 1>)> {
        let sample = NutritionValue::create_random_value();
        let mut rng = rand::thread_rng();
        let random_template = self
            .template_filenames
            .choose(&mut rng)
            .ok_or(anyhow::Error::msg("failed to choose one template"))?;
        let mut context = tera::Context::new();
        context.insert("sample", &sample);
        let svg_doc = self.tera.render(random_template, &context)?;
        let (pixmap, w, h) = get_svg_pixmap(svg_doc, self.root_dir.to_owned())?;
        let data = Data::<u8, 3>::new(pixmap, Shape::new([w, h, 4])).convert();
        // resvg saves as 4 color channel. in our case, we pretty sure, it is not relevant.
        // So we remove whole dimension and permute tensor to comply with common cases
        let xt = Tensor::from_data_device(data, &device);
        let xt = xt // w, h, c
            .swap_dims(0, 2) // c, h, w
            .swap_dims(1, 2)
            / 255.0; // c, w, h
                     //let xt = xt.to_device(tch::Device::Mps);

        let expected_result = sample.to_tensor(device);

        Ok((xt, expected_result.float()))
    }
}

fn read_templates(templates_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let templates: Vec<PathBuf> = fs::read_dir(templates_dir)?
        .filter_map(|entry| {
            if entry
                .as_ref()
                .ok()
                .and_then(|item| item.metadata().ok())
                .map(|meta| meta.is_file())
                .is_some_and(|is_file| is_file)
            {
                entry.ok()
            } else {
                None
            }
        })
        .map(|entry| entry.path())
        .collect();

    Ok(templates)
}

fn get_svg_pixmap(
    svg_data: String,
    resources_dir: PathBuf,
) -> anyhow::Result<(Vec<u8>, usize, usize)> {
    let svg_tree = {
        let opt = usvg::Options {
            resources_dir: Some(resources_dir),
            ..Default::default()
        };
        // Get file's absolute directory.
        let mut font_db = fontdb::Database::new();
        font_db.load_system_fonts();

        let mut tree = usvg::Tree::from_data(svg_data.as_bytes(), &opt).unwrap();
        tree.convert_text(&font_db);
        resvg::Tree::from_usvg(&tree)
    };
    let pixmap_size = svg_tree.size.to_int_size();
    let mut pixmap = tiny_skia::Pixmap::new(pixmap_size.width(), pixmap_size.height()).unwrap();
    svg_tree.render(tiny_skia::Transform::default(), &mut pixmap.as_mut());
    let w = pixmap.width() as usize;
    let h = pixmap.height() as usize;

    Ok((pixmap.take(), w, h))
}
