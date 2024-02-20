mod dataset_loader;
mod layer;
// mod my_module;
use mlp_rust::my_module;
use layer::Layer;
//visualize the first image
use image::{ImageBuffer, Rgb};
use ndarray::prelude::*;
use ndarray::Zip;
fn main() {
    // Deconstruct the returned Mnist struct.
    let (train_data, train_labels, _testdata, _testlabels) = dataset_loader::get_data_from_mnist_files();

    let mut layer1 = Layer::new(10, 784, my_module::relu);
    let mut layer2 = Layer::new(10, 10, softmax);

    //print shape of input
    println!(
        "Input shape: {:?}, {:?}",
        train_data.shape(),
        train_labels.shape()
    );

    let offset = 0;
    let batch_size = 1000;

    for _epoch in 0..9 {
        //println!("Epoch: {:?}", epoch);
        for i in 0..49 {
            let inputs = train_data.slice(s![0*batch_size..0*batch_size + batch_size, ..]).t().to_owned();
            //println!("Inputs shape: {:?}", inputs.shape());

            let labels = train_labels.slice(s![0*batch_size..0*batch_size + batch_size, ..]).to_owned();

            let output1 = layer1.weighted_sum(&inputs);
            let activated_output1 = layer1.activation_fn(&output1);

            let output2 = layer2.weighted_sum(&activated_output1);
            let activated_output2 = layer2.activation_fn(&output2);

            //transform the labels to one-hot encoding
            let mut one_hot_labels = Array2::zeros((batch_size, 10));
            for i in 0..batch_size {
                let label = labels[[i, 0]] as usize;
                one_hot_labels
                    .slice_mut(s![i, ..])
                    .assign(&my_module::one_hot_encode(label));
            }

            let dz2 = &activated_output2 - &one_hot_labels.t();

            let w2_delta = (1.0/batch_size as f32) * &dz2.dot(&activated_output1.t());
            let b2_delta = dz2.sum_axis(Axis(1)) / batch_size as f32;

            let activated_output1_derivative = my_module::get_relu_derivative(&output1);

            let dz1 = layer2.w_matrix.t().dot(&dz2) * &activated_output1_derivative;

            let w1_delta = dz1.dot(&inputs.t()) / batch_size as f32;
            let b1_delta = dz1.sum_axis(Axis(1)) / batch_size as f32;


            layer2.w_matrix = layer2.w_matrix - 0.01 * &w2_delta;
            layer2._b_matrix =  layer2._b_matrix - &b2_delta * 0.01;

            layer1.w_matrix = layer1.w_matrix - 0.01 * &w1_delta;
            layer1._b_matrix = layer1._b_matrix - &b1_delta * 0.01;
            // offset += 500;

            let error_magnitude = dz2.iter().fold(0.0, |acc, x| acc + x.powi(2));
            println!("error: {:?}", error_magnitude);
            let w2_magnitude = layer2.w_matrix.iter().fold(0.0, |acc, x| acc + x);
            println!("w2_magnitude: {:?}", w2_magnitude);


            // offset += 500;
            test_current_model(&activated_output2, &one_hot_labels.t().to_owned());
        }
    }
}

fn test_current_model(activated_output2: &Array2<f32>, one_hot_labels: &Array2<f32>) {
    //Compare the output of layer2, highest value is the predicted label, compare with the actual label
    // //println!("Activated_output2: {:?}", activated_output2.shape());
    // //println!("First_1000_labels: {:?}", one_hot_labels.shape());
    let mut correct_count = 0;
    for i in 0..activated_output2.shape()[1] {
        let mut max_index = 0;
        let mut max_value = 0.0;
        for j in 0..activated_output2.shape()[0] {
            if activated_output2[[j, i]] > max_value {
                max_value = activated_output2[[j, i]];
                max_index = j;
            }
        }
        if one_hot_labels[[max_index, i]] == 1.0 {
            correct_count += 1;
        }
    }
    println!("Correct count: {:?}", correct_count);

}

fn _mean_squared_error(predicted: &Array2<f32>, actual: &Array2<f32>) -> Array2<f32> {
    let mut output = Array2::zeros(predicted.raw_dim());
    for i in 0..predicted.shape()[0] {
        for j in 0..predicted.shape()[1] {
            output[[i, j]] = (predicted[[i, j]] - actual[[i, j]]).powi(2);
        }
    }
    return output;
}

fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let mut output = Array2::zeros(x.raw_dim());

    // Iterate over each row of the input matrix
    x.outer_iter().into_iter().enumerate().for_each(|(i, row)| {
        // Find the maximum value in the row
        let max = row.fold(f32::NEG_INFINITY, |acc, &elem| elem.max(acc));

        // Calculate the sum of exponentials while subtracting the maximum value (to avoid overflow)
        let sum: f32 = row.map(|&elem| (elem - max).exp()).sum();

        // Calculate the softmax values for the row and store them in the output matrix
        Zip::from(output.row_mut(i)).and(&row).apply(|output_elem, &x_elem| {
            *output_elem = (x_elem - max).exp() / sum;
        });
    });

    output
}

