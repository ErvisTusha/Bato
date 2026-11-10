//! Original content-adaptive selection of independent byte representations.
//! Provenance: LZ77, canonical Huffman and reversible delta/shuffle descriptions
//! referenced in the sibling implementations. Select actual complete bytes,
//! never an extension, MIME type, random sample or an estimated entropy score.
use super::{filters, huffman, lz, numbers, varint};
use crate::core::{
    frame::block_cost,
    memory::try_copy,
    types::{BatoError, BlockType, TransformId},
};

pub(super) struct Candidate {
    pub kind: BlockType,
    pub transform: TransformId,
    pub payload: Vec<u8>,
    pub length: usize,
}
impl Candidate {
    pub fn cost(&self) -> usize {
        block_cost(self.kind, self.length, self.payload.len())
    }
    fn consider(&mut self, kind: BlockType, transform: TransformId, payload: Vec<u8>) {
        if payload.len() >= self.length {
            return;
        }
        let cost = block_cost(kind, self.length, payload.len());
        if cost < self.cost() {
            self.kind = kind;
            self.transform = transform;
            self.payload = payload;
        }
    }
    fn evaluate(&mut self, raw: &[u8], transform: TransformId) -> Result<(), BatoError> {
        if raw.len() > 1 && raw.iter().all(|&b| b == raw[0]) {
            self.consider(BlockType::Repeat, transform, vec![raw[0]]);
            return Ok(());
        }
        let copies = lz::encode(raw)?;
        if let Some(coded) = huffman::encode(
            &copies,
            self.payload
                .len()
                .saturating_sub(varint::len(copies.len() as u64) + 1),
        ) {
            let mut combined = Vec::new();
            varint::push(&mut combined, copies.len());
            combined.extend_from_slice(&coded);
            self.consider(BlockType::LzHuffman, transform, combined);
        }
        self.consider(BlockType::Lz, transform, copies);
        if let Some(coded) = huffman::encode(raw, self.payload.len().saturating_sub(1)) {
            self.consider(BlockType::Huffman, transform, coded);
        }
        Ok(())
    }
}
pub(super) fn unfiltered(raw: &[u8]) -> Result<Candidate, BatoError> {
    let mut best = Candidate {
        kind: BlockType::Stored,
        transform: TransformId::None,
        payload: try_copy(raw)?,
        length: raw.len(),
    };
    best.evaluate(raw, TransformId::None)?;
    Ok(best)
}
pub(super) fn best(raw: &[u8], x86: Option<TransformId>) -> Result<Candidate, BatoError> {
    let mut best = Candidate {
        kind: BlockType::Stored,
        transform: TransformId::None,
        payload: try_copy(raw)?,
        length: raw.len(),
    };
    best.evaluate(raw, TransformId::None)?;
    // No representation can have less than one payload byte. Stop on the
    // exact minimum; this also avoids pointless transforms of uniform data.
    if best.kind == BlockType::Repeat {
        return Ok(best);
    }
    for filter in filters::CANDIDATES.into_iter().chain(x86) {
        let transformed = filters::forward(raw, filter)?;
        best.evaluate(&transformed, filter)?;
    }
    if let Some(coded) = numbers::encode(raw)? {
        best.consider(BlockType::Numeric, TransformId::None, coded);
    }
    Ok(best)
}
pub(super) fn decode(kind: BlockType, payload: &[u8], length: usize) -> Result<Vec<u8>, BatoError> {
    match kind {
        BlockType::Numeric => numbers::decode(payload, length),
        BlockType::Stored if payload.len() == length => try_copy(payload),
        BlockType::Repeat if payload.len() == 1 => {
            crate::core::memory::try_filled(length, payload[0])
        }
        BlockType::Lz => lz::decode(payload, length),
        BlockType::Huffman => huffman::decode(payload, length),
        BlockType::LzHuffman => {
            let mut at = 0;
            let n = varint::take(payload, &mut at)?;
            if n == 0 || n > length.saturating_add(10) {
                return Err(BatoError::CorruptStream(
                    "invalid token stream length".into(),
                ));
            }
            let tokens = huffman::decode(&payload[at..], n)?;
            lz::decode(&tokens, length)
        }
        _ => Err(BatoError::CorruptStream(
            "invalid independent segment".into(),
        )),
    }
}
