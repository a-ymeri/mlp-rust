#[cfg(test)]
mod tests {
    use mlp_rust::my_module;
    #[test]
    fn test_one_hot_encode() {
        for i in 0..10 {
            let mut predicted_one_hot = my_module::one_hot_encode(i);
            let mut actual_one_hot = ndarray::arr1(&[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]);
            actual_one_hot[i] = 1.0;
            assert_eq!(predicted_one_hot, actual_one_hot);
        }
    }

    #[test]
    fn test_relu(){
        let mut input = ndarray::arr2(&[[1.0, 2.0, 3.0], [-4.0, 5.0, 0.0]]);
        let mut output = my_module::relu(&input);
        let mut expected_output = ndarray::arr2(&[[1.0, 2.0, 3.0], [0.0, 5.0, 0.0]]);
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_relu_derivative(){
        let mut input = ndarray::arr2(&[[1.0, 2.0, 3.0], [-4.0, 5.0, 0.0]]);
        let mut output = my_module::get_relu_derivative(&input);
        let mut expected_output = ndarray::arr2(&[[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]);
        assert_eq!(output, expected_output);

    }
}