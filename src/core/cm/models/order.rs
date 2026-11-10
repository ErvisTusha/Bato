//! Original universal byte-order contexts for the CM ensemble.
//! Provenance: conventional finite byte contexts, independently implemented.
use super::super::hash;
/// Byte orders modeled by hashed contexts; order 5 lives in the text expert.
const ORDERS: [usize; 6] = [0, 1, 2, 3, 4, 6];
pub(in crate::core::cm) const COUNT: usize = ORDERS.len();
pub(in crate::core::cm) fn contexts(history: &[u8]) -> [u64; COUNT] {
    ORDERS.map(|order| hash::bytes(history, order))
}
