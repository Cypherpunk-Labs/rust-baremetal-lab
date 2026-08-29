use alloc::string::String;
use alloc::vec::Vec;

const MAGIC: u32 = 0x544F4B45;

/// Byte-level GPT-2 BPE tokenizer, decoded from the compact blob produced by
/// `tokenizer-builder`. Correct for normal text (words, single spaces,
/// punctuation, digits); the exact pre-tokenizer split for multiple leading
/// spaces is not replicated (a minor, rare edge case).
pub struct Tokenizer {
    byte_to_id: [u32; 256],
    vocab_bytes: Vec<Vec<u8>>,
    merges: Vec<(u32, u32, u32)>, // (left, right, merged)
}

impl Tokenizer {
    /// Parse the embedded blob into a usable tokenizer.
    pub fn new(blob: &'static [u8]) -> Self {
        let mut off = 0usize;
        let rd = |off: &mut usize, n: usize| -> &[u8] {
            let s = &blob[*off..*off + n];
            *off += n;
            s
        };
        let rd_u32 = |off: &mut usize| -> u32 {
            let b: [u8; 4] = rd(off, 4).try_into().unwrap();
            u32::from_le_bytes(b)
        };
        assert_eq!(rd_u32(&mut off), MAGIC, "bad tokenizer magic");
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

        let mut merges = Vec::with_capacity(merge_len);
        for _ in 0..merge_len {
            let l = rd_u32(&mut off);
            let r = rd_u32(&mut off);
            let m = rd_u32(&mut off);
            merges.push((l, r, m));
        }
        assert_eq!(off, blob.len(), "tokenizer blob not fully consumed");

        Tokenizer { byte_to_id, vocab_bytes, merges }
    }

    /// Encode a UTF-8 string into token ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = text
            .as_bytes()
            .iter()
            .map(|&b| self.byte_to_id[b as usize])
            .collect();

        loop {
            let mut best: Option<(usize, u32)> = None; // (idx, merged)
            let mut best_rank = u32::MAX;
            for (i, (l, r, m)) in self.merges.iter().enumerate() {
                // early-skip: check if this pair appears at/before current best
                if (i as u32) >= best_rank {
                    continue;
                }
                // find first occurrence of this pair in ids
                for j in 0..ids.len().saturating_sub(1) {
                    if ids[j] == *l && ids[j + 1] == *r {
                        best_rank = i as u32;
                        best = Some((j, *m));
                        break;
                    }
                }
            }
            match best {
                Some((j, merged)) => {
                    ids[j] = merged;
                    ids.remove(j + 1);
                }
                None => break,
            }
        }
        ids
    }

    /// Decode token ids back into a UTF-8 string (lossy).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.vocab_bytes[id as usize]);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
