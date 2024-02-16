mod dataset_loader;
mod layer;
use layer::Layer;
//visualize the first image
use image::{ImageBuffer, Rgb};
use ndarray::prelude::*;
use ndarray::{s, Array3};
fn main() {
    // Deconstruct the returned Mnist struct.
    let (train_data, train_labels, _, _) = dataset_loader::get_data_from_mnist_files();

    let mut layer1 = Layer::new(10, 784, relu);
    let mut layer2 = Layer::new(10, 10, softmax);

    //print shape of input
    println!(
        "Input shape: {:?}, {:?}",
        train_data.shape(),
        train_labels.shape()
    );

    let mut offset = 0;

    for epoch in 0..10 {
        for i in 0..101 {
            let mut first_1000_inputs = train_data.slice(s![offset..offset + 500, ..]).to_owned();

            let mut first_1000_labels = train_labels.slice(s![offset..offset + 500, ..]).to_owned();

            let mut output1 = layer1.weighted_sum(&first_1000_inputs);
            let mut activated_output1 = layer1.activation_fn(&output1);

            println!("Activated output1: {:?}", activated_output1.sum());

            let mut output2 = layer2.weighted_sum(&activated_output1);
            let mut activated_output2 = layer2.activation_fn(&output2);

            //transform the labels to one-hot encoding
            let mut one_hot_labels = Array2::zeros((first_1000_labels.shape()[0], 10));
            for i in 0..first_1000_labels.shape()[0] {
                let label = first_1000_labels[[i, 0]] as usize;
                one_hot_labels
                    .slice_mut(s![i, ..])
                    .assign(&one_hot_encode(label));
            }

            //calculate the loss
            // println!("Loss: {:?}",);
            let error = &activated_output2 - &one_hot_labels;

            let w2_delta = &activated_output1.dot(&error.t()) / first_1000_inputs.shape()[0] as f32;
            let b2_delta = error.sum_axis(Axis(0)) / first_1000_inputs.shape()[0] as f32;

            let error2 = error.dot(&layer2.w_matrix.t());
            let activated_output1_derivative = activated_output1.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

            layer2.w_matrix -= &w2_delta;
            layer2._b_matrix -= &b2_delta;


            offset += 500;
        }
    }
}

fn relu(x: &Array2<f32>) -> Array2<f32> {
    let mut output = Array2::zeros(x.raw_dim());
    for i in 0..x.shape()[0] {
        for j in 0..x.shape()[1] {
            output[[i, j]] = f32::max(0.0, x[[i, j]]);
        }
    }
    return output;
}

fn one_hot_encode(label: usize) -> Array1<f32> {
    let mut output = Array1::zeros(10);
    output[label] = 1.0;
    return output;
}

fn mean_squared_error(y_pred: &Array2<f32>, y_true: &Array2<f32>) -> f32 {
    let mut sum = 0.0;
    for i in 0..y_pred.shape()[0] {
        for j in 0..y_pred.shape()[1] {
            sum += (y_pred[[i, j]] - y_true[[i, j]]).powi(2);
        }
    }
    return sum / (y_pred.shape()[0] * y_pred.shape()[1]) as f32;
}

fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let mut output = Array2::zeros(x.raw_dim());
    for i in 0..x.shape()[0] {
        let mut sum = 0.0;
        for j in 0..x.shape()[1] {
            sum += f32::exp(x[[i, j]]);
        }
        //if sum is inf, panic
        if (sum.is_infinite()) {
            panic!("Sum is infinite");
        }
        // println!("Sum: {:?}", sum);
        for j in 0..x.shape()[1] {
            output[[i, j]] = f32::exp(x[[i, j]]) / sum;
        }
    }
    return output;
}

fn _output_image(train_data: Array3<f32>, index: usize) {
    let image = train_data.slice(s![index, .., ..]);

    let mut img = ImageBuffer::new(28, 28);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let val = image[[y as usize, x as usize]] * 255.0;
        *pixel = Rgb([val as u8, val as u8, val as u8]);
    }
}
