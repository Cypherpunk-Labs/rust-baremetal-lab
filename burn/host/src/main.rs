use burn::{
    module::Module,
    record::{NoStdInferenceRecorder, Recorder},
    tensor::{Int, Shape, Tensor, TensorData},
};
use burn_flex::Flex;
use kernel::model::{SmolLmConfig, SmolLmModel, SmolLmModelRecord};
use kernel::tokenizer::Tokenizer;
use std::io::{BufRead, Write};

type Backend = Flex;

const VOCAB: usize = 49152;
const TEMP: f32 = 0.8;
const TOP_K: usize = 40;
const MAX_NEW: usize = 64;

// ---------------------------------------------------------------------------
// Sampling helpers (duplicated from the kernel's main.rs so the kernel source
// stays untouched; identical logic/constants).
// ---------------------------------------------------------------------------

/// exp(x) — the host binary uses std, but keep this identical to the kernel's
/// no_std Taylor-series implementation so results match bit-for-bit.
fn expf(x: f32) -> f32 {
    let mut sum = 1.0f32;
    let mut term = 1.0f32;
    let mut n = 1.0f32;
    for _ in 0..14 {
        term *= x / n;
        sum += term;
        n += 1.0;
    }
    sum
}

/// Deterministic PRNG (xorshift64*) seeded from the system counter, matching
/// the kernel.
struct Rng(u64);

impl Rng {
    fn new() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Rng(nanos | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// Sample a token from logits with temperature + top-k + categorical sampling.
fn sample_token(logits: &[f32], rng: &mut Rng, temperature: f32, top_k: usize) -> u32 {
    let t = temperature.max(1e-4);
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let k = top_k.min(idx.len());
    let mut probs: Vec<f32> = idx[..k].iter().map(|&i| expf(logits[i] / t)).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (j, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return idx[j] as u32;
        }
    }
    idx[0] as u32
}

// ---------------------------------------------------------------------------
// Model / tokenizer loading (mirrors kernel startup).
// ---------------------------------------------------------------------------

/// Resolve the directory holding the `.bin` data files. Defaults to the repo's
/// `kernel/src/`, but can be overridden with `HOST_DATA_DIR` (e.g. when running
/// inside a Linux guest where the files live under a 9p mount).
fn data_dir() -> String {
    std::env::var("HOST_DATA_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../kernel/src/").to_string())
}

fn load_bytes(rel: &str) -> &'static [u8] {
    let dir = data_dir();
    let dir = if dir.ends_with('/') { dir } else { format!("{dir}/") };
    let path = format!("{dir}{rel}");
    let data = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("failed to read {}: {} (run `make model` / `make tokenizer`?)", path, e)
    });
    Box::leak(data.into_boxed_slice())
}

fn load_model() -> SmolLmModel<Backend> {
    let device = Default::default();
    let config = SmolLmConfig::default();
    let bytes = load_bytes("smollm-135m.bin");
    println!("[Host] Read {} bytes", bytes.len());

    let recorder = NoStdInferenceRecorder::new();
    let record: SmolLmModelRecord<Backend> = recorder.load(bytes, &device).unwrap();
    let model = SmolLmModel::<Backend>::new(&config, &device).load_record(record);
    println!("[Host] Model loaded onto burn-flex backend");
    model
}

// ---------------------------------------------------------------------------
// Greedy benchmark (KV cache), byte-for-byte same format as the kernel.
// ---------------------------------------------------------------------------

fn greedy_benchmark(model: &SmolLmModel<Backend>, steps: usize) -> (Vec<i64>, f64) {
    let device = Default::default();
    let mut cache = model.new_cache();

    println!("[Host] Starting autoregressive generation loop (greedy, KV cache)...");
    let mut total_ms: f64 = 0.0;

    // Initial prompt: process all prompt tokens, cache their K/V.
    let prompt: Vec<i64> = vec![15496, 11];
    let seq_len = prompt.len();
    let mut next_token: i64;
    {
        let data = TensorData::new(prompt.clone(), Shape::new([1, seq_len]));
        let input_tensor = Tensor::<Backend, 2, Int>::from_data(data, &device);

        let t0 = std::time::Instant::now();
        let logits = model.forward_step(input_tensor, &mut cache);
        let last_logits = logits.slice([0..1, seq_len - 1..seq_len]);
        next_token = last_logits.argmax(2).into_scalar() as i64;
        let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_ms += step_ms;
        println!("[step 0] token={} ({:.0} ms)", next_token, step_ms);
    }

    // Subsequent steps: feed only the previous token; attention reads cached K/V.
    let mut tokens: Vec<i64> = vec![15496, 11, next_token];
    for step in 1..steps {
        let data = TensorData::new(vec![next_token], Shape::new([1, 1]));
        let input_tensor = Tensor::<Backend, 2, Int>::from_data(data, &device);

        let t0 = std::time::Instant::now();
        let logits = model.forward_step(input_tensor, &mut cache);
        let last_logits = logits.slice([0..1, 0..1]);
        next_token = last_logits.argmax(2).into_scalar() as i64;
        let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_ms += step_ms;
        tokens.push(next_token);
        println!("[step {}] token={} ({:.0} ms)", step, next_token, step_ms);
    }

    let tps = (steps as f64 * 1000.0) / total_ms;
    (tokens, tps)
}

// ---------------------------------------------------------------------------
// Chat loop (sampling), same format as the kernel.
// ---------------------------------------------------------------------------

fn chat_loop(model: &SmolLmModel<Backend>) {
    let tokenizer = Tokenizer::new(load_bytes("tokenizer.bin"));
    println!(
        "[Chat] tokenizer ready (vocab={}), enter a prompt and press Enter:",
        VOCAB
    );

    let stdin = std::io::stdin();
    loop {
        print!("\n[Chat] Prompt> ");
        std::io::stdout().flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break; // EOF
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }

        println!("\n[Chat] Encoded {} chars", line.chars().count());
        let ids = tokenizer.encode(line);
        if ids.is_empty() {
            println!("[Chat] empty tokenization");
            continue;
        }
        println!("[Chat] {} tokens", ids.len());

        let device = Default::default();
        let mut cache = model.new_cache();
        let mut all_ids: Vec<i64> = ids.iter().map(|&i| i as i64).collect();
        let mut next_token: i64;
        let mut rng = Rng::new();
        let mut gen_count: u64 = 0;

        // Prompt step: process all prompt tokens, cache K/V, sample first token.
        {
            let seq_len = all_ids.len();
            let data = TensorData::new(all_ids.clone(), Shape::new([1, seq_len]));
            let input_tensor = Tensor::<Backend, 2, Int>::from_data(data, &device);
            let logits = model.forward_step(input_tensor, &mut cache);
            let last = logits.slice([0..1, seq_len - 1..seq_len]);
            let last_data = last.into_data().to_vec::<f32>().unwrap();
            next_token = sample_token(&last_data, &mut rng, TEMP, TOP_K) as i64;
        }
        println!("[Chat] <|im_start|>");
        let t0 = std::time::Instant::now();
        for _ in 0..MAX_NEW {
            let piece = tokenizer.decode(&[next_token as u32]);
            print!("{}", piece);
            std::io::stdout().flush().ok();
            all_ids.push(next_token);
            gen_count += 1;

            if next_token == 0 {
                break; // <|endoftext|>
            }

            let data = TensorData::new(vec![next_token], Shape::new([1, 1]));
            let input_tensor = Tensor::<Backend, 2, Int>::from_data(data, &device);
            let logits = model.forward_step(input_tensor, &mut cache);
            let last = logits.slice([0..1, 0..1]);
            let last_data = last.into_data().to_vec::<f32>().unwrap();
            next_token = sample_token(&last_data, &mut rng, TEMP, TOP_K) as i64;
        }
        let gen_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let tps = (gen_count as f64 * 1000.0) / gen_ms;
        println!(
            "\n[Chat] done ({} tokens, {:.2} ms/token, {:.3} tokens/sec)",
            all_ids.len(),
            gen_ms / gen_count as f64,
            tps
        );
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bench = args.iter().any(|a| a == "--bench");

    let model = load_model();

    if bench {
        // Sequential benchmark: run the greedy benchmark N times and average TPS.
        let n = args
            .iter()
            .position(|a| a == "--bench")
            .and_then(|p| args.get(p + 1))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(5);
        println!("\n[Host] Sequential benchmark: {} runs", n);
        let mut tps_sum = 0.0f64;
        let mut tokens: Option<Vec<i64>> = None;
        for i in 0..n {
            println!("\n===== run {} =====", i + 1);
            let (tok, tps) = greedy_benchmark(&model, 10);
            tokens = Some(tok);
            println!("[Host] run {} TPS: {:.3}", i + 1, tps);
            tps_sum += tps;
        }
        let avg = tps_sum / n as f64;
        println!("\n[Host] avg TPS over {} runs: {:.3}", n, avg);
        println!("[Host] Final tokens: {:?}", tokens.unwrap());
        return;
    }

    greedy_benchmark(&model, 10);
    println!("[Host] Inference test completed successfully");
    chat_loop(&model);
}
