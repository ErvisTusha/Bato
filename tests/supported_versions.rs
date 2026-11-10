use bato_engine::{decompress, read_stream_header, BatoError, VERSION_MAJOR, VERSION_MINOR};
use std::io::Cursor;

#[test]
fn unsupported_stream_versions_are_rejected_before_payload_decode() {
    for minor in (0..=u8::MAX).filter(|&minor| minor != VERSION_MINOR) {
        let mut header = [b'B', b'A', b'T', b'O', VERSION_MAJOR, VERSION_MINOR, 0];
        header[5] = minor;
        assert!(matches!(read_stream_header(&mut Cursor::new(header)),
            Err(BatoError::UnsupportedVersion { major, minor: actual }) if major == VERSION_MAJOR && actual == minor));
        assert!(matches!(decompress(&header),
            Err(BatoError::UnsupportedVersion { major, minor: actual }) if major == VERSION_MAJOR && actual == minor));
    }
}

#[test]
fn foreign_stream_magic_is_rejected() {
    let header = *b"BTS6\x06\x13\x0b";
    assert!(matches!(read_stream_header(&mut Cursor::new(header)),
        Err(BatoError::InvalidMagic(magic)) if magic == *b"BTS6"));
    assert!(matches!(decompress(&header),
        Err(BatoError::InvalidMagic(magic)) if magic == *b"BTS6"));
}
