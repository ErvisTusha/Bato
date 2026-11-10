//! Segment geometry recorded in the stream header: the only encoder settings
//! the decoder needs. Three bytes: segment log2, chunk log2, table-cap log2.

use crate::core::types::BatoError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Descriptor {
    pub(crate) segment_log2: u8,
    pub(crate) chunk_log2: u8,
    pub(crate) table_log2: u8,
}

impl Default for Descriptor {
    fn default() -> Self {
        Self {
            segment_log2: 26,
            chunk_log2: 18,
            table_log2: 22,
        }
    }
}

impl Descriptor {
    pub(crate) const BYTES: usize = 3;

    pub(crate) fn validate(self) -> Result<(), BatoError> {
        if !(16..=30).contains(&self.segment_log2)
            || !(12..=20).contains(&self.chunk_log2)
            || self.chunk_log2 > self.segment_log2
            || self.table_log2 > 24
        {
            return Err(BatoError::InvalidOptions("segment log2 must be 16..30, chunk log2 12..20 and no larger than the segment, and table cap log2 0..24".into()));
        }
        Ok(())
    }

    pub(crate) fn segment_bytes(self) -> usize {
        1usize << self.segment_log2
    }
    pub(crate) fn chunk_bytes(self) -> usize {
        1usize << self.chunk_log2
    }

    pub(crate) fn encode(self) -> Result<[u8; Self::BYTES], BatoError> {
        self.validate()?;
        Ok([self.segment_log2, self.chunk_log2, self.table_log2])
    }

    pub(crate) fn decode(bytes: [u8; Self::BYTES]) -> Result<Self, BatoError> {
        let descriptor = Self {
            segment_log2: bytes[0],
            chunk_log2: bytes[1],
            table_log2: bytes[2],
        };
        descriptor.validate().map_err(|_| {
            BatoError::CorruptStream("invalid segment geometry or table cap".into())
        })?;
        Ok(descriptor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_roundtrips_and_rejects_invalid_geometry() {
        let descriptor = Descriptor::default();
        assert_eq!(descriptor.encode().unwrap(), [26, 18, 22]);
        assert_eq!(
            Descriptor::decode(descriptor.encode().unwrap()).unwrap(),
            descriptor
        );
        for bytes in [
            [15, 12, 22],
            [31, 18, 22],
            [26, 11, 22],
            [26, 21, 22],
            [16, 17, 22],
            [26, 18, 25],
        ] {
            assert!(Descriptor::decode(bytes).is_err());
        }
    }

    #[test]
    fn geometry_bounds_are_checked_before_shifts_or_allocation() {
        for segment in 0..=255u8 {
            for chunk in 0..=255u8 {
                let descriptor = Descriptor {
                    segment_log2: segment,
                    chunk_log2: chunk,
                    table_log2: 20,
                };
                let valid =
                    (16..=30).contains(&segment) && (12..=20).contains(&chunk) && chunk <= segment;
                assert_eq!(descriptor.validate().is_ok(), valid);
            }
        }
        for cap in 0..=255u8 {
            let descriptor = Descriptor {
                table_log2: cap,
                ..Default::default()
            };
            assert_eq!(descriptor.validate().is_ok(), cap <= 24);
        }
    }
}
