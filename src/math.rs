use super::activations::{
    NeuronActivation,
    LayerActivation,
};

pub fn apply_neuron_activation_f32(v: f32, activation: &NeuronActivation) -> f32 {
    match activation {
        NeuronActivation::LeakyReLU(alpha) => {
            if v > 0.0 {
                v
            } else {
                v * *alpha
            }
        },
    }
}

pub fn apply_layer_activation_f32(values: &Vec<f32>, activation: &LayerActivation) -> Vec<f32> {
    match activation {
        LayerActivation::SoftMax => {
            let mut res = Vec::new();

            let mut sum = 0.0;

            for &v in values.iter() {
                sum += v.exp();
            }

            for &v in values.iter() {
                res.push(v.exp() / sum);
            }

            res
        },
    }
}

pub fn index_of_max_value(values: &Vec<f32>) -> usize {
    let mut max_index = 0;
    let mut max_value = values[0];

    for (i, &v) in values.iter().enumerate() {
        if v > max_value {
            max_index = i;
            max_value = v;
        }
    }

    max_index
}

mod tests {
    use super::*;

    #[test]
    fn test_index_of_max_value() {
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(index_of_max_value(&values), 4);
    }
}
