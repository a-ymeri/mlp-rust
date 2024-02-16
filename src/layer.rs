use ndarray::Array;
use ndarray::Array1;
use ndarray::Array2;
use rand::Rng;
pub struct Layer {
    pub w_matrix: Array2<f32>,
    pub _b_matrix: Array1<f32>,
    pub activation_fn: fn(&Array2<f32>) -> Array2<f32>,
}

impl Layer {
    // pub fn new(neurons: usize, dimensions: usize, act_fn: fn(&Vec<f64>) -> Vec<f64>) -> Layer {
    //     Layer {
    //         w_matrix: vec![vec![1.0; dimensions + 1]; neurons],
    //         activation_fn: act_fn,
    //     }
    // }

    pub fn new(
        neurons: usize,
        dimensions: usize,
        act_fn: fn(&Array2<f32>) -> Array2<f32>,
    ) -> Layer {
        let mut w_matrix: Array2<f32> = Array::zeros((neurons, dimensions));
        let mut rng = rand::thread_rng();
        for i in 0..neurons {
            for j in 0..dimensions {
                w_matrix[[i, j]] = rng.gen_range(-1.0, 1.0);
            }
        }
        Layer {
            w_matrix: w_matrix,
            _b_matrix: Array1::zeros(neurons),
            activation_fn: act_fn,
        }
    }

    pub fn activation_fn(&self, input: &Array2<f32>) -> Array2<f32> {
        return (self.activation_fn)(input);
    }

    pub fn weighted_sum(&self, input: &Array2<f32>) -> Array2<f32> {
        let neuron_num = self.w_matrix.shape()[0]; //number of neurons
        let sample_count = input.shape()[0]; //number of inputs
        let dimension_count = input.shape()[1]; //number of features
        let mut output = vec![0.0; neuron_num * sample_count];
        for i in 0..neuron_num {
            for j in 0..sample_count {
                let mut sum = 0.0;
                for k in 0..dimension_count {
                    sum += input[[j, k]] * self.w_matrix[[i, k]];
                }
                output[i * sample_count + j] = sum;
            }
        }
        //take into account the bias
        return Array::from_shape_vec((sample_count, neuron_num), output).unwrap();
    }
}
