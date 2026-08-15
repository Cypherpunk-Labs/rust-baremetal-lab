//! Convert SmolLM GPT-2 byte-level BPE tokenizer.json into a compact binary
//! blob for the no_std kernel.
//!
//! Blob layout (little-endian):
//!   u32 magic     = 0x544F4B45 ('TOKE')
//!   u32 vocab_len
//!   u32 merge_len
//!   u32 byte_to_id[256]                single-byte token id per raw byte (0xFFFFFFFF = none)
//!   vocab entries: for each id, u32 len + raw bytes (byte-decoded token string)
//!   merges: for each of merge_len, u32 left_id + u32 right_id + u32 merged_id (rank order)
//!
//! Usage: tokenizer-builder <tokenizer.json> <out.bin>

use serde_json::Value;
use std::collections::HashMap;
use std::io::Write;

const MAGIC: u32 = 0x544F4B45; // 'TOKE'

fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = (b'!'..=b'~')
        .chain(b'\xa1'..=b'\xac')
        .chain(b'\xae'..=b'\xff')
        .collect();
    let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();
    let mut n: u32 = 0;
    for b in 0..=255u32 {
        if !bs.contains(&(b as u8)) {
            bs.push(b as u8);
            cs.push(char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }
    bs.into_iter().zip(cs).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() == 3, "usage: tokenizer-builder <in.json> <out.bin>");
    let src = std::fs::read_to_string(&args[1]).expect("read tokenizer.json");
    let v: Value = serde_json::from_str(&src).expect("parse tokenizer.json");

    let vocab = v["model"]["vocab"].as_object().expect("vocab object");
    let merges = v["model"]["merges"].as_array().expect("merges array");

    // Build id -> key (char-string) and byte-decoded bytes.
    let vocab_len = vocab.len();
    let mut id_to_key: Vec<Option<&str>> = vec![None; vocab_len];
    for (k, val) in vocab {
        let id = val.as_u64().expect("vocab id") as usize;
        id_to_key[id] = Some(k.as_str());
    }

    let byte_enc = bytes_to_unicode();
    let byte_dec: HashMap<char, u8> = byte_enc.iter().map(|(&b, &c)| (c, b)).collect();
    fn char_to_bytes(s: &str, dec: &HashMap<char, u8>) -> Vec<u8> {
        s.chars().map(|c| dec[&c]).collect()
    }

    // byte -> token id
    let mut byte_to_id = [0xFFFF_FFFFu32; 256];
    let mut missing = Vec::new();
    for b in 0..=255u8 {
        let key: String = byte_enc[&b].to_string();
        if let Some(id) = vocab.get(key.as_str()) {
            byte_to_id[b as usize] = id.as_u64().unwrap() as u32;
        } else {
            missing.push(b);
        }
    }
    eprintln!("note: {} bytes lack a standalone token: {:?}", missing.len(), missing);

    // id -> raw bytes
    let vocab_bytes: Vec<Vec<u8>> = (0..vocab_len)
        .map(|i| char_to_bytes(id_to_key[i].expect("id present"), &byte_dec))
        .collect();

    // merges -> (left_id, right_id, merged_id)
    let mut merge_triples: Vec<(u32, u32, u32)> = Vec::with_capacity(merges.len());
    for m in merges {
        let s = m.as_str().expect("merge string");
        let mut it = s.split(' ');
        let left = it.next().unwrap();
        let right = it.next().unwrap();
        let left_id = vocab[left].as_u64().unwrap() as u32;
        let right_id = vocab[right].as_u64().unwrap() as u32;
        // merged token bytes = left bytes ++ right bytes; find its id in vocab.
        let mut merged_bytes = char_to_bytes(left, &byte_dec);
        merged_bytes.extend_from_slice(&char_to_bytes(right, &byte_dec));
        let merged_id = (0..vocab_len)
            .find(|&i| vocab_bytes[i] == merged_bytes)
            .expect("merged token in vocab") as u32;
        merge_triples.push((left_id, right_id, merged_id));
    }

    // Write blob.
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&(vocab_len as u32).to_le_bytes());
    out.extend_from_slice(&(merge_triples.len() as u32).to_le_bytes());
    for b in byte_to_id {
        out.extend_from_slice(&b.to_le_bytes());
    }
    for bytes in &vocab_bytes {
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    for (l, r, m) in &merge_triples {
        out.extend_from_slice(&l.to_le_bytes());
        out.extend_from_slice(&r.to_le_bytes());
        out.extend_from_slice(&m.to_le_bytes());
    }

    let mut f = std::fs::File::create(&args[2]).expect("create out file");
    f.write_all(&out).expect("write out file");
    println!(
        "Wrote {} bytes (vocab={}, merges={}) to {}",
        out.len(),
        vocab_len,
        merge_triples.len(),
        args[2]
    );
}
