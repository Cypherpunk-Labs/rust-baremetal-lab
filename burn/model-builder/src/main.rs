use burn::{
    module::{Module, Param},
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::{Tensor, TensorData},
};
use burn_flex::Flex;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::fs::File;
use std::io::Write as _;

use kernel::model::{SmolLmConfig, SmolLmModel};

fn load_f32(t: &TensorView) -> Vec<f32> {
    assert_eq!(t.dtype(), Dtype::F32, "unexpected dtype");
    let bytes = t.data();
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load_linear(
    st: &SafeTensors,
    name: &str,
    device: &<Flex as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<Flex, 2> {
    let t = st.tensor(name).unwrap();
    let shape: Vec<usize> = t.shape().to_vec();
    assert_eq!(shape.len(), 2, "{} is not a matrix", name);
    let (rows, cols) = (shape[0], shape[1]);
    let data = load_f32(&t);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    Tensor::from_data(TensorData::new(out, [cols, rows]), device)
}

fn load_vec(st: &SafeTensors, name: &str, device: &<Flex as burn::tensor::backend::BackendTypes>::Device) -> Tensor<Flex, 1> {
    let t = st.tensor(name).unwrap();
    let data = load_f32(&t);
    Tensor::from_data(TensorData::new(data, t.shape().to_vec()), device)
}

fn load_embedding(
    st: &SafeTensors,
    name: &str,
    device: &<Flex as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<Flex, 2> {
    let t = st.tensor(name).unwrap();
    let shape: Vec<usize> = t.shape().to_vec();
    assert_eq!(shape.len(), 2, "{} is not a matrix", name);
    let data = load_f32(&t);
    Tensor::from_data(TensorData::new(data, shape), device)
}

fn main() {
    let device = Default::default();
    let config = SmolLmConfig::default();
    let mut model = SmolLmModel::<Flex>::new(&config, &device);

    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/model.safetensors");
    let buffer = std::fs::read(model_path).unwrap();
    let st = SafeTensors::deserialize(&buffer).unwrap();

    model.embed_tokens.weight =
        Param::from_tensor(load_embedding(&st, "model.embed_tokens.weight", &device));

    for layer_idx in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{}.", layer_idx);

        model.layers[layer_idx].input_layernorm.gamma =
            Param::from_tensor(load_vec(&st, &format!("{}input_layernorm.weight", prefix), &device));
        model.layers[layer_idx].post_attention_layernorm.gamma = Param::from_tensor(load_vec(
            &st,
            &format!("{}post_attention_layernorm.weight", prefix),
            &device,
        ));

        model.layers[layer_idx].self_attn.q_proj.weight = Param::from_tensor(load_linear(
            &st,
            &format!("{}self_attn.q_proj.weight", prefix),
            &device,
        ));
        model.layers[layer_idx].self_attn.k_proj.weight = Param::from_tensor(load_linear(
            &st,
            &format!("{}self_attn.k_proj.weight", prefix),
            &device,
        ));
        model.layers[layer_idx].self_attn.v_proj.weight = Param::from_tensor(load_linear(
            &st,
            &format!("{}self_attn.v_proj.weight", prefix),
            &device,
        ));
        model.layers[layer_idx].self_attn.o_proj.weight = Param::from_tensor(load_linear(
            &st,
            &format!("{}self_attn.o_proj.weight", prefix),
            &device,
        ));

        model.layers[layer_idx].mlp.gate_proj.weight = Param::from_tensor(load_linear(
            &st,
            &format!("{}mlp.gate_proj.weight", prefix),
            &device,
        ));
        model.layers[layer_idx].mlp.up_proj.weight = Param::from_tensor(load_linear(
            &st,
            &format!("{}mlp.up_proj.weight", prefix),
            &device,
        ));
        model.layers[layer_idx].mlp.down_proj.weight = Param::from_tensor(load_linear(
            &st,
            &format!("{}mlp.down_proj.weight", prefix),
            &device,
        ));

        if layer_idx % 10 == 0 {
            println!("Converted layer {}", layer_idx);
        }
    }

    model.norm.gamma = Param::from_tensor(load_vec(&st, "model.norm.weight", &device));

    let record = model.into_record();
    let bytes: Vec<u8> = BinBytesRecorder::<FullPrecisionSettings>::new()
        .record(record, ())
        .unwrap();

    let out_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../kernel/src/smollm-135m.bin");
    let mut f = File::create(out_path).unwrap();
    f.write_all(&bytes).unwrap();
    println!(
        "Wrote {} bytes ({} tensors) to {}",
        bytes.len(),
        st.names().len(),
        out_path
    );
}