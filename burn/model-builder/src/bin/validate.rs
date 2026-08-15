use burn::{
    module::Module,
    record::{NoStdInferenceRecorder, Recorder},
    tensor::{Int, Shape, Tensor, TensorData},
};
use burn_flex::Flex;
use kernel::model::{SmolLmConfig, SmolLmModel, SmolLmModelRecord};

fn main() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../kernel/src/smollm-135m.bin");
    let bytes: &'static [u8] = Box::leak(std::fs::read(path).unwrap().into_boxed_slice());
    println!("Read {} bytes", bytes.len());

    let device = Default::default();
    let config = SmolLmConfig::default();

    let t0 = std::time::Instant::now();
    let recorder = NoStdInferenceRecorder::new();
    let record: SmolLmModelRecord<Flex> = recorder.load(bytes, &device).unwrap();
    println!("Record loaded in {:?}", t0.elapsed());

    let t1 = std::time::Instant::now();
    let model = SmolLmModel::<Flex>::new(&config, &device).load_record(record);
    println!("Model loaded in {:?}", t1.elapsed());

    let mut input_ids: Vec<i64> = vec![15496, 11];
    let mut total = std::time::Duration::ZERO;
    for step in 0..10 {
        let seq_len = input_ids.len();
        let data = TensorData::new(input_ids.clone(), Shape::new([1, seq_len]));
        let input_tensor = Tensor::<Flex, 2, Int>::from_data(data, &device);

        let t2 = std::time::Instant::now();
        let logits = model.forward(input_tensor);
        let last_logits = logits.slice([0..1, seq_len - 1..seq_len]);
        let next_token = last_logits.argmax(2).into_scalar() as i64;
        total += t2.elapsed();
        input_ids.push(next_token);
        println!("[step {}] token={} ({:?})", step, next_token, t2.elapsed());
    }
    println!("Final tokens: {:?}", input_ids);
    println!(
        "avg {:.2} ms/token, {:.3} tokens/sec",
        total.as_secs_f64() * 1000.0 / 10.0,
        10.0 / total.as_secs_f64()
    );
}