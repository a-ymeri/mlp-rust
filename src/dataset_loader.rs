use mnist::*;
use ndarray::prelude::*;
use ndarray::Array2;
use ndarray::Array3;
use image::{ImageBuffer, Rgb};

pub fn get_data_from_mnist_files() -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array2::from_shape_vec((50_000, 28 * 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);

    // // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);

        let test_data = Array2::from_shape_vec((10_000, 28 * 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);

    // // Convert the returned Mnist struct to Array2 format
    let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    // let _test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
    //     .expect("Error converting images to Array3 struct")
    //     .map(|x| *x as f32 / 256.);

    // let _test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
    //     .expect("Error converting testing labels to Array2 struct")
    //     .map(|x| *x as f32);

    return (train_data, train_labels, test_data, test_labels);
}



fn _output_image(train_data: Array2<f32>, index: usize) {
    let image = train_data.slice(s![index, ..]).into_shape((28, 28)).unwrap();

    let mut img = ImageBuffer::new(28, 28);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let val = image[[y as usize, x as usize]] * 255.0;
        *pixel = Rgb([val as u8, val as u8, val as u8]);
    }
    let filename = format!("img_{}.png", index);
    img.save(filename).unwrap();
}