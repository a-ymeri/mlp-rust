pub mod my_module {
    use ndarray::prelude::*;
    use ndarray::Array1;
    pub fn one_hot_encode(label: usize) -> Array1<f32> {
        let mut output = Array1::zeros(10);
        output[label] = 1.0;
        return output;
    }

    pub fn relu(x: &Array2<f32>) -> Array2<f32> {
        let mut output = Array2::zeros(x.raw_dim());
        for i in 0..x.shape()[0] {
            for j in 0..x.shape()[1] {
                output[[i, j]] = f32::max(0.0, x[[i, j]]);
            }
        }
        return output;
    }

    pub fn get_relu_derivative(x: &Array2<f32>) -> Array2<f32> {
        let mut output = Array2::zeros(x.raw_dim());
        for i in 0..x.shape()[0] {
            for j in 0..x.shape()[1] {
                if x[[i, j]] > 0.0 {
                    output[[i, j]] = 1.0;
                }
            }
        }
        return output;
    }
}