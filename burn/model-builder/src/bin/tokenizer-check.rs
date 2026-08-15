//! Host-side reference for the byte-level BPE tokenizer, reading the same blob
//! the kernel uses. Validates encode/decode against known token outputs.

use std::collections::HashMap;

const MAGIC: u32 = 0x544F4B45;

struct Tokenizer {
    byte_to_id: [u32; 256],
    vocab_bytes: Vec<Vec<u8>>,
    // pair (left,right) -> (rank, merged_id)
    ranks: HashMap<(u32, u32), (u32, u32)>,
}

impl Tokenizer {
    fn from_blob(blob: &[u8]) -> Self {
        let mut off = 0;
        let rd = |off: &mut usize, n: usize| -> &[u8] {
            let s = &blob[*off..*off + n];
            *off += n;
            s
        };
        let rd_u32 = |off: &mut usize| -> u32 {
            let b: [u8; 4] = rd(off, 4).try_into().unwrap();
            u32::from_le_bytes(b)
        };
        assert_eq!(rd_u32(&mut off), MAGIC, "bad magic");
        let vocab_len = rd_u32(&mut off) as usize;
        let merge_len = rd_u32(&mut off) as usize;

        let mut byte_to_id = [0xFFFF_FFFFu32; 256];
        for b in 0..256 {
            byte_to_id[b] = rd_u32(&mut off);
        }

        let mut vocab_bytes = Vec::with_capacity(vocab_len);
        for _ in 0..vocab_len {
            let l = rd_u32(&mut off) as usize;
            vocab_bytes.push(rd(&mut off, l).to_vec());
        }

        let mut ranks = HashMap::with_capacity(merge_len);
        for rank in 0..merge_len {
            let l = rd_u32(&mut off);
            let r = rd_u32(&mut off);
            let m = rd_u32(&mut off);
            ranks.insert((l, r), (rank as u32, m));
        }
        assert_eq!(off, blob.len(), "blob not fully consumed");

        Tokenizer { byte_to_id, vocab_bytes, ranks }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut ids: Vec<u32> = bytes
            .iter()
            .map(|&b| {
                let t = self.byte_to_id[b as usize];
                assert_ne!(t, 0xFFFF_FFFF, "no token for byte 0x{:02x}", b);
                t
            })
            .collect();

        loop {
            // find lowest-rank adjacent pair
            let mut best: Option<(usize, u32, u32)> = None; // (idx, rank, merged)
            let mut best_rank = u32::MAX;
            for i in 0..ids.len().saturating_sub(1) {
                if let Some(&(rank, merged)) = self.ranks.get(&(ids[i], ids[i + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best = Some((i, rank, merged));
                    }
                }
            }
            match best {
                Some((i, _rank, merged)) => {
                    ids[i] = merged;
                    ids.remove(i + 1);
                }
                None => break,
            }
        }
        ids
    }

    fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| self.vocab_bytes[id as usize].clone()).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn main() {
    let blob = std::fs::read(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../kernel/src/tokenizer.bin"),
    )
    .expect("read tokenizer.bin");
    let tok = Tokenizer::from_blob(&blob);

    let cases: &[(&str, &[u32])] = &[
        ("Hello", &[19556]),
        ("Hello world", &[19556, 905]),
        ("The quick brown fox", &[504, 2365, 6354, 16438]),
        ("  spaced  text", &[216, 23861, 216, 1694]),
        ("a", &[81]),
        ("hello", &[28120]),
        ("spaced text", &[2401, 2568, 1694]),
        ("Hello, world!", &[19556, 28, 905, 17]),
        ("123 abc", &[33, 34, 35, 43239]),
        ("foo bar baz", &[15236, 2753, 278, 1437]),
        (" spaced", &[23861]),
        ("  spaced", &[216, 23861]),
        ("   spaced", &[256, 23861]),
        ("spaced  text", &[2401, 2568, 216, 1694]),
        ("a b", &[81, 278]),
        ("  a", &[216, 253]),
        ("Hi  there", &[26843, 216, 665]),
    ];

    let mut ok = true;
    for (text, expect) in cases {
        let ids = tok.encode(text);
        let pass = ids == *expect;
        ok &= pass;
        let dec = tok.decode(&ids);
        println!(
            "{:?} -> {:?} (expect {:?}) {} decode={:?}",
            text,
            ids,
            expect,
            if pass { "OK" } else { "FAIL" },
            dec
        );
    }
    println!("all pass: {}", ok);
}
