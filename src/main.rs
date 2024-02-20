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
    let learning_rate = 0.1;
    for _epoch in 0..30 {
        //println!("Epoch: {:?}", epoch);
        let mut activated_output2 = Array2::zeros((10, batch_size));
        let mut one_hot_labels = Array2::zeros((10, batch_size));
        for i in 0..49 {
            let inputs = train_data.slice(s![i*batch_size..i*batch_size + batch_size, ..]).t().to_owned();
            //println!("Inputs shape: {:?}", inputs.shape());



            let labels = train_labels.slice(s![i*batch_size..i*batch_size + batch_size, ..]).to_owned();

            let output1 = layer1.w_matrix.dot(&inputs) + &layer1._b_matrix;
            let activated_output1 = my_module::relu(&output1);

            let output2 = layer2.w_matrix.dot(&activated_output1) + &layer2._b_matrix;
            activated_output2 = softmax(&output2);

            /*
                Backpropagation
             */
            
            //transform the labels to one-hot encoding
            one_hot_labels = Array2::zeros((10, batch_size));
            for i in 0..batch_size {
                let label = labels[[i, 0]] as usize;
                one_hot_labels[[label, i]] = 1.0;
            }

            let dz2 = &activated_output2 - &one_hot_labels;

            let w2_delta = (1.0/batch_size as f32) * &dz2.dot(&activated_output1.t());
            let b2_delta = (1.0/batch_size as f32) * dz2.sum_axis(Axis(1)).insert_axis(Axis(1));

            let output1_derivative = my_module::get_relu_derivative(&output1);

            let dz1 = layer2.w_matrix.t().dot(&dz2) * &output1_derivative;

            let w1_delta = dz1.dot(&inputs.t()) / batch_size as f32;
            let b1_delta = dz1.sum_axis(Axis(1)).insert_axis(Axis(1)) / batch_size as f32;


            layer2.w_matrix = layer2.w_matrix - learning_rate * &w2_delta;
            layer2._b_matrix = layer2._b_matrix - &b2_delta * learning_rate;

            layer1.w_matrix = layer1.w_matrix - learning_rate * &w1_delta;
            layer1._b_matrix = layer1._b_matrix - &b1_delta * learning_rate;
            // offset += 500;

            // let error_magnitude = dz2.iter().fold(0.0, |acc, x| acc + x.powi(2));
            // println!("error: {:?}", error_magnitude);
            // let w2_magnitude = layer2.w_matrix.iter().fold(0.0, |acc, x| acc + x);
            // println!("w2_magnitude: {:?}", w2_magnitude);


            // offset += 500;
        }
        test_current_model(&activated_output2, &one_hot_labels);
    }

    //test against test data
    let test_data = _testdata.t().to_owned();
    let test_labels = _testlabels.to_owned();
    let output1 = layer1.w_matrix.dot(&test_data) + &layer1._b_matrix;
    let activated_output1 = my_module::relu(&output1);

    let output2 = layer2.w_matrix.dot(&activated_output1) + &layer2._b_matrix;
    let activated_output2 = softmax(&output2);

    let mut labels_encoded = Array2::zeros((10, test_labels.shape()[0]));
    for i in 0..test_labels.shape()[0] {
        let label = test_labels[[i, 0]] as usize;
        labels_encoded[[label, i]] = 1.0;
    }
    test_current_model(&activated_output2, &labels_encoded);
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
    // println!("Softmax input: {:?}", x.shape());

    let mut aug_x = x.clone();
    for i in 0..x.shape()[1] {
        let mut sum = 0.0;
        for j in 0..x.shape()[0] {
            sum += f32::exp(x[[j, i]]);
        }
        for j in 0..x.shape()[0] {
            aug_x[[j, i]] = f32::exp(x[[j, i]]) / sum;
        }
    }
    return aug_x.to_owned();
}

