//! Striped Reed-Solomon parity with bounded per-stripe memory.
use super::*;
use reed_solomon_erasure::galois_8::ReedSolomon;

const STRIPE: usize = 16 * 1024;

pub(super) fn codec(data: u64, parity: u8) -> Result<ReedSolomon, BatoError> {
    if !(2..=16).contains(&parity) || data + u64::from(parity) > 256 {
        return Err(invalid(
            "multi-parity requires 2–16 parity volumes and at most 256 total volumes",
        ));
    }
    ReedSolomon::new(data as usize, usize::from(parity)).map_err(|error| invalid(error.to_string()))
}

fn parity_name(index: usize) -> String {
    format!("parity-{:02}", index + 1)
}

fn read_stripe(file: &mut File, stripe: &mut [u8], remaining: u64) -> io::Result<()> {
    stripe.fill(0);
    let size = (stripe.len() as u64).min(remaining) as usize;
    file.read_exact(&mut stripe[..size])
}

pub(super) fn write(
    directory: &Directory,
    limit: u64,
    bytes: u64,
    parity: u8,
) -> Result<Vec<[u8; 32]>, BatoError> {
    let count = count_parts(limit, bytes)?;
    let codec = codec(count, parity)?;
    let mut inputs = Vec::new();
    let mut output = Vec::new();
    let mut hashes = Vec::new();
    let mut parity_hashes = vec![Sha256::new(); usize::from(parity)];
    for index in 1..=count {
        let mut file = directory.open_file(Path::new(&part_name(index)))?;
        hashes.push(crate::core::jobs::fingerprint(&mut file)?);
        inputs.push(file);
    }
    for index in 0..usize::from(parity) {
        output.push(directory.create_file(Path::new(&parity_name(index)))?);
    }
    let width = limit.min(bytes);
    let mut position = 0;
    while position < width {
        let length = (width - position).min(STRIPE as u64) as usize;
        let mut shards = vec![vec![0; length]; count as usize + usize::from(parity)];
        for (index, file) in inputs.iter_mut().enumerate() {
            let size = (bytes - index as u64 * limit).min(limit);
            read_stripe(file, &mut shards[index], size.saturating_sub(position))?;
        }
        codec
            .encode(&mut shards)
            .map_err(|error| invalid(error.to_string()))?;
        for ((file, digest), shard) in output
            .iter_mut()
            .zip(&mut parity_hashes)
            .zip(&shards[count as usize..])
        {
            file.write_all(shard)?;
            digest.update(shard);
        }
        position += length as u64;
    }
    for (file, digest) in output.into_iter().zip(parity_hashes) {
        file.sync_all()?;
        hashes.push(digest.finalize().into());
    }
    Ok(hashes)
}

fn verified(
    directory: &Directory,
    name: &str,
    size: u64,
    digest: [u8; 32],
) -> Result<Option<File>, BatoError> {
    let mut file = match directory.open_file(Path::new(name)) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if file.metadata()?.len() != size || crate::core::jobs::fingerprint(&mut file)? != digest {
        return Ok(None);
    }
    Ok(Some(file))
}

pub(super) fn join(
    volumes: &Volumes,
    directory: &Directory,
    hashes: &[[u8; 32]],
) -> Result<File, BatoError> {
    let count = count_parts(volumes.limit, volumes.bytes)? as usize;
    let parity = usize::from(volumes.parity);
    let codec = codec(count as u64, volumes.parity)?;
    let width = volumes.limit.min(volumes.bytes);
    let mut files = Vec::new();
    for (index, &digest) in hashes.iter().enumerate() {
        let (name, size) = if index < count {
            (
                part_name(index as u64 + 1),
                volumes.part_size(index as u64 + 1),
            )
        } else {
            (parity_name(index - count), width)
        };
        files.push(verified(directory, &name, size, digest)?);
    }
    let missing: Vec<_> = (0..count).filter(|&index| files[index].is_none()).collect();
    if !missing.is_empty() {
        if files.iter().filter(|file| file.is_none()).count() > parity {
            return Err(invalid(
                "not enough intact volumes for parity reconstruction",
            ));
        }
        let mut recovered = missing
            .iter()
            .map(|_| tempfile::tempfile())
            .collect::<io::Result<Vec<_>>>()?;
        let mut position = 0;
        while position < width {
            let length = (width - position).min(STRIPE as u64) as usize;
            let mut shards = Vec::with_capacity(files.len());
            for (index, file) in files.iter_mut().enumerate() {
                let shard = if let Some(file) = file {
                    let size = if index < count {
                        volumes.part_size(index as u64 + 1)
                    } else {
                        width
                    };
                    let mut stripe = vec![0; length];
                    read_stripe(file, &mut stripe, size.saturating_sub(position))?;
                    Some(stripe)
                } else {
                    None
                };
                shards.push(shard);
            }
            codec
                .reconstruct_data(&mut shards)
                .map_err(|error| invalid(error.to_string()))?;
            for (&index, output) in missing.iter().zip(&mut recovered) {
                let size = volumes.part_size(index as u64 + 1);
                let length = size.saturating_sub(position).min(length as u64) as usize;
                let stripe = shards[index]
                    .as_ref()
                    .ok_or_else(|| invalid("missing recovered stripe"))?;
                output.write_all(&stripe[..length])?;
            }
            position += length as u64;
        }
        for (&index, mut file) in missing.iter().zip(recovered) {
            if crate::core::jobs::fingerprint(&mut file)? != hashes[index] {
                return Err(invalid("reconstructed volume failed verification"));
            }
            files[index] = Some(file);
        }
    }
    let mut output = BufWriter::new(tempfile::tempfile()?);
    let digest = copy_parts(
        &mut output,
        volumes.limit,
        volumes.bytes,
        |index, size, sink| {
            let mut file = files[index as usize - 1]
                .take()
                .ok_or_else(|| invalid("missing data volume"))?;
            file.rewind()?;
            transfer(&mut file, sink, size)?;
            require_eof(&mut file)
        },
    )?;
    if digest != volumes.digest {
        return Err(invalid("assembled archive failed SHA-256 verification"));
    }
    let mut file = output.into_inner().map_err(|error| error.into_error())?;
    file.rewind()?;
    Ok(file)
}
