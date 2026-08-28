//! Minimal safetensors reader (header JSON + raw tensor byte ranges).
//! Only the features needed for SmolLM-135M F32 weights are implemented.

use alloc::string::String;
use alloc::vec::Vec;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DType {
    F32,
    F16,
    Other,
}

pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Byte range within the post-header data section.
    pub start: usize,
    pub end: usize,
}

pub struct SafeTensors<'a> {
    tensors: Vec<TensorInfo>,
    data: &'a [u8],
}

impl<'a> SafeTensors<'a> {
    pub fn parse(buf: &'a [u8]) -> Result<SafeTensors<'a>, &'static str> {
        if buf.len() < 8 {
            return Err("buffer too small for safetensors header length");
        }
        let header_len = u64::from_le_bytes([buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]]) as usize;
        let header_end = 8 + header_len;
        if header_end > buf.len() {
            return Err("safetensors header length out of range");
        }
        let header = core::str::from_utf8(&buf[8..header_end]).map_err(|_| "header not utf8")?;
        let json = parse_json(header)?;
        let obj = match json {
            Json::Obj(o) => o,
            _ => return Err("top-level header is not an object"),
        };
        let mut tensors = Vec::new();
        for (key, val) in obj {
            if key == "__metadata__" {
                continue;
            }
            let fields = match val {
                Json::Obj(f) => f,
                _ => return Err("tensor entry is not an object"),
            };
            let mut dtype = DType::Other;
            let mut shape = Vec::new();
            let mut start = 0usize;
            let mut end = 0usize;
            for (k, v) in fields {
                match k {
                    "dtype" => {
                        dtype = match v {
                            Json::Str(s) if s == "F32" => DType::F32,
                            Json::Str(s) if s == "F16" => DType::F16,
                            _ => DType::Other,
                        }
                    }
                    "shape" => {
                        if let Json::Arr(a) = v {
                            for d in a {
                                if let Json::Num(n) = d {
                                    shape.push(n as usize);
                                }
                            }
                        }
                    }
                    "data_offsets" => {
                        if let Json::Arr(a) = v {
                            if a.len() == 2 {
                                if let (Json::Num(s), Json::Num(e)) = (&a[0], &a[1]) {
                                    start = *s as usize;
                                    end = *e as usize;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            tensors.push(TensorInfo { name: String::from(key), dtype, shape, start, end });
        }
        let data = &buf[header_end..];
        Ok(SafeTensors { tensors, data })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn raw(&self, name: &str) -> Option<&'a [u8]> {
        for t in &self.tensors {
            if t.name == name {
                return Some(&self.data[t.start..t.end]);
            }
        }
        None
    }

    /// F32 view of a tensor (panics if missing or not F32). Returns an owned
    /// `Vec<f32>` so callers don't depend on the buffer being 4-byte aligned
    /// (embedded `include_bytes!` data is not guaranteed aligned).
    pub fn tensor_f32(&self, name: &str) -> Vec<f32> {
        let t = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("missing tensor {}", name));
        assert_eq!(t.dtype, DType::F32, "tensor {} is not F32", name);
        bytes_to_f32_vec(&self.data[t.start..t.end])
    }
}

/// Reinterpret little-endian F32 bytes as `f32` values (copy, alignment-safe).
pub fn bytes_to_f32_vec(b: &[u8]) -> Vec<f32> {
    assert_eq!(b.len() % 4, 0, "f32 buffer not a multiple of 4 bytes");
    let mut out = Vec::with_capacity(b.len() / 4);
    for chunk in b.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

// --- tiny JSON parser (objects, arrays, strings, numbers, bool, null) -------

enum Json<'a> {
    Null,
    Bool(bool),
    Num(f64),
    Str(&'a str),
    Arr(Vec<Json<'a>>),
    Obj(Vec<(&'a str, Json<'a>)>),
}

fn skip_ws(s: &str, pos: usize) -> usize {
    let b = s.as_bytes();
    let mut p = pos;
    while p < b.len() && (b[p] == b' ' || b[p] == b'\n' || b[p] == b'\r' || b[p] == b'\t') {
        p += 1;
    }
    p
}

fn parse_json(s: &str) -> Result<Json, &'static str> {
    let (v, p) = parse_value(s, 0)?;
    let p = skip_ws(s, p);
    if p != s.len() {
        return Err("trailing characters after JSON value");
    }
    Ok(v)
}

fn parse_value(s: &str, pos: usize) -> Result<(Json, usize), &'static str> {
    let p = skip_ws(s, pos);
    let b = s.as_bytes();
    if p >= b.len() {
        return Err("unexpected end of JSON");
    }
    match b[p] {
        b'{' => parse_object(s, p),
        b'[' => parse_array(s, p),
        b'"' => {
            let (st, np) = parse_string(s, p)?;
            Ok((Json::Str(st), np))
        }
        b't' | b'f' => {
            if s[p..].starts_with("true") {
                Ok((Json::Bool(true), p + 4))
            } else if s[p..].starts_with("false") {
                Ok((Json::Bool(false), p + 5))
            } else {
                Err("invalid literal")
            }
        }
        b'n' => {
            if s[p..].starts_with("null") {
                Ok((Json::Null, p + 4))
            } else {
                Err("invalid literal")
            }
        }
        b'-' | (b'0'..=b'9') => parse_number(s, p),
        _ => Err("unexpected JSON token"),
    }
}

fn parse_string(s: &str, pos: usize) -> Result<(&str, usize), &'static str> {
    let b = s.as_bytes();
    let mut p = pos + 1;
    while p < b.len() {
        match b[p] {
            b'"' => {
                let st = &s[pos + 1..p];
                return Ok((st, p + 1));
            }
            b'\\' => p += 2, // skip escaped char
            _ => p += 1,
        }
    }
    Err("unterminated string")
}

fn parse_number(s: &str, pos: usize) -> Result<(Json, usize), &'static str> {
    let b = s.as_bytes();
    let mut p = pos;
    if b[p] == b'-' {
        p += 1;
    }
    while p < b.len() && (b[p].is_ascii_digit() || b[p] == b'.' || b[p] == b'e' || b[p] == b'E' || b[p] == b'+' || b[p] == b'-') {
        p += 1;
    }
    let num_str = &s[pos..p];
    let v: f64 = num_str.parse().map_err(|_| "invalid number")?;
    Ok((Json::Num(v), p))
}

fn parse_array(s: &str, pos: usize) -> Result<(Json, usize), &'static str> {
    let b = s.as_bytes();
    let mut p = pos + 1;
    let mut items = Vec::new();
    p = skip_ws(s, p);
    if p < b.len() && b[p] == b']' {
        return Ok((Json::Arr(items), p + 1));
    }
    loop {
        let (v, np) = parse_value(s, p)?;
        items.push(v);
        p = skip_ws(s, np);
        if p >= b.len() {
            return Err("unterminated array");
        }
        if b[p] == b',' {
            p += 1;
        } else if b[p] == b']' {
            return Ok((Json::Arr(items), p + 1));
        } else {
            return Err("expected ',' or ']' in array");
        }
    }
}

fn parse_object(s: &str, pos: usize) -> Result<(Json, usize), &'static str> {
    let b = s.as_bytes();
    let mut p = pos + 1;
    let mut fields = Vec::new();
    p = skip_ws(s, p);
    if p < b.len() && b[p] == b'}' {
        return Ok((Json::Obj(fields), p + 1));
    }
    loop {
        p = skip_ws(s, p);
        if p >= b.len() || b[p] != b'"' {
            return Err("expected string key in object");
        }
        let (k, np) = parse_string(s, p)?;
        p = skip_ws(s, np);
        if p >= b.len() || b[p] != b':' {
            return Err("expected ':' in object");
        }
        let (v, np2) = parse_value(s, p + 1)?;
        fields.push((k, v));
        p = skip_ws(s, np2);
        if p >= b.len() {
            return Err("unterminated object");
        }
        if b[p] == b',' {
            p += 1;
        } else if b[p] == b'}' {
            return Ok((Json::Obj(fields), p + 1));
        } else {
            return Err("expected ',' or '}' in object");
        }
    }
}
