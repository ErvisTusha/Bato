// Original lossless fragment pilot using the LZ77 distance/length principle.
// Source ranges name an earlier native archive member, regardless of segment.
use std::{
    collections::HashMap,
    env, fs,
    io::{self, ErrorKind, Read},
    path::{Component, Path, PathBuf},
};

const LIMIT: usize = 8 * 1024 * 1024;
const MIN_COPY: usize = 192;
const MAX_RUNS: usize = 1 << 14;

type Spot = (usize, usize);
type Command = [u32; 4];

fn invalid() -> io::Error {
    io::Error::new(ErrorKind::InvalidData, "invalid fragment catalogue")
}

fn walk(
    root: &Path,
    at: &Path,
    dirs: &mut Vec<PathBuf>,
    files: &mut Vec<PathBuf>,
) -> io::Result<()> {
    for entry in fs::read_dir(root.join(at))? {
        let entry = entry?;
        let child = at.join(entry.file_name());
        let kind = entry.file_type()?;
        if kind.is_dir() {
            dirs.push(child.clone());
            walk(root, &child, dirs, files)?;
        } else if kind.is_file() {
            files.push(child);
        } else {
            return Err(invalid());
        }
    }
    Ok(())
}

fn inventory(root: &Path) -> io::Result<(Vec<PathBuf>, Vec<PathBuf>)> {
    let (mut dirs, mut files) = (Vec::new(), Vec::new());
    walk(root, Path::new(""), &mut dirs, &mut files)?;
    dirs.sort();
    files.sort();
    Ok((dirs, files))
}

fn hash(data: &[u8]) -> u64 {
    let value = u64::from_le_bytes(data.try_into().unwrap());
    let mixed = (value ^ (value >> 33)).wrapping_mul(0xff51_afd7_ed55_8ccd);
    mixed ^ (mixed >> 33)
}

fn anchor(data: &[u8], offset: usize) -> Option<u64> {
    let word = data.get(offset..offset + 8)?;
    let key = hash(word);
    (key & 31 == 0).then_some(key)
}

fn matched(a: &[u8], b: &[u8]) -> usize {
    let limit = a.len().min(b.len());
    let mut length = 0;
    while length + 8 <= limit {
        let x = u64::from_le_bytes(a[length..length + 8].try_into().unwrap());
        let y = u64::from_le_bytes(b[length..length + 8].try_into().unwrap());
        let diff = x ^ y;
        if diff != 0 {
            return length + (diff.trailing_zeros() / 8) as usize;
        }
        length += 8;
    }
    length
        + a[length..limit]
            .iter()
            .zip(&b[length..limit])
            .take_while(|(x, y)| x == y)
            .count()
}

fn factor(
    raw: &[u8],
    previous: &[Vec<u8>],
    index: &HashMap<u64, [Option<Spot>; 2]>,
) -> (Vec<u8>, Vec<Command>) {
    let (mut literals, mut commands) = (Vec::new(), Vec::new());
    let (mut at, mut copied) = (0, 0);
    while at + MIN_COPY <= raw.len() && commands.len() < MAX_RUNS {
        let mut best = None;
        let mut length = MIN_COPY - 1;
        if let Some(spots) = anchor(raw, at).and_then(|key| index.get(&key)) {
            for &(file, offset) in spots.iter().flatten() {
                let n = matched(&previous[file][offset..], &raw[at..]);
                if n > length {
                    length = n;
                    best = Some((file, offset));
                }
            }
        }
        if let Some((file, mut offset)) = best {
            let mut start = at;
            while start > copied && offset > 0 && raw[start - 1] == previous[file][offset - 1] {
                start -= 1;
                offset -= 1;
            }
            let literal_len = start - copied;
            literals.extend_from_slice(&raw[copied..start]);
            copied = at + length;
            commands.push([
                literal_len as u32,
                file as u32,
                offset as u32,
                (copied - start) as u32,
            ]);
            at = copied;
        } else {
            at += 1;
        }
    }
    literals.extend_from_slice(&raw[copied..]);
    (literals, commands)
}

fn push(out: &mut Vec<u8>, mut value: usize) {
    while value >= 128 {
        out.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    out.push(value as u8);
}

fn relative(path: &Path) -> io::Result<&str> {
    if path.as_os_str().is_empty() || !path.components().all(|c| matches!(c, Component::Normal(_)))
    {
        return Err(invalid());
    }
    path.to_str().ok_or_else(invalid)
}

fn read_bounded(path: &Path, limit: usize) -> io::Result<Vec<u8>> {
    let mut data = Vec::new();
    fs::File::open(path)?
        .take((limit + 1) as u64)
        .read_to_end(&mut data)?;
    if data.len() > limit {
        return Err(invalid());
    }
    Ok(data)
}

fn prepare(root: &Path, stage: &Path) -> io::Result<()> {
    let (dirs, files) = inventory(root)?;
    let payload = stage.join("payload");
    let mut total = 0usize;
    for path in &files {
        let length =
            usize::try_from(fs::metadata(root.join(path))?.len()).map_err(|_| invalid())?;
        total = total.saturating_add(length);
        if total > LIMIT || files.len() > 4096 {
            return Ok(());
        }
    }
    let mut prior = Vec::<Vec<u8>>::new();
    let mut encoded = Vec::new();
    let mut index = HashMap::<u64, [Option<Spot>; 2]>::new();
    let mut recipes = Vec::new();
    let mut count = 0;
    let mut remaining = LIMIT;
    for (id, path) in files.iter().enumerate() {
        let raw = read_bounded(&root.join(path), remaining)?;
        remaining -= raw.len();
        let duplicate = prior.iter().position(|other| other == &raw);
        let (literals, commands) = if duplicate.is_none() {
            factor(&raw, &prior, &index)
        } else {
            (Vec::new(), Vec::new())
        };
        let use_recipe =
            literals.len() + 16 * commands.len() + 24 < raw.len() * 3 / 4 && !commands.is_empty();
        if use_recipe {
            count += 1;
            push(&mut recipes, id);
            push(&mut recipes, raw.len());
            push(&mut recipes, commands.len());
            for command in commands {
                for value in command {
                    push(&mut recipes, value as usize);
                }
            }
            encoded.push(Some(literals));
        } else {
            encoded.push(None);
        }
        if duplicate.is_none() {
            for offset in 0..raw.len().saturating_sub(MIN_COPY - 1) {
                if let Some(key) = anchor(&raw, offset) {
                    let spots = index.entry(key).or_default();
                    *spots = [Some((id, offset)), spots[0]];
                }
            }
        }
        prior.push(raw);
    }
    if count == 0 {
        return Ok(());
    }
    fs::create_dir_all(&payload)?;
    for dir in dirs {
        fs::create_dir_all(payload.join(dir))?;
    }
    let mut manifest = Vec::new();
    push(&mut manifest, files.len());
    for (id, path) in files.iter().enumerate() {
        relative(path)?;
        fs::write(
            payload.join(path),
            encoded[id].as_deref().unwrap_or(&prior[id]),
        )?;
    }
    push(&mut manifest, count);
    manifest.extend(recipes);
    fs::write(stage.join("recipes"), manifest)
}

fn take<'a>(input: &mut &'a [u8], size: usize) -> io::Result<&'a [u8]> {
    let (value, rest) = input.split_at_checked(size).ok_or_else(invalid)?;
    *input = rest;
    Ok(value)
}

fn number(input: &mut &[u8]) -> io::Result<usize> {
    let mut value = 0;
    for shift in [0, 7, 14, 21, 28] {
        let byte = take(input, 1)?[0];
        if shift == 28 && byte > 15 {
            return Err(invalid());
        }
        value |= usize::from(byte & 127) << shift;
        if byte < 128 {
            return if byte == 0 && shift != 0 {
                Err(invalid())
            } else {
                Ok(value)
            };
        }
    }
    Err(invalid())
}

fn rebuild(input: &mut &[u8], literals: &[u8], prior: &[Vec<u8>]) -> io::Result<Vec<u8>> {
    let length = number(input)?;
    let count = number(input)?;
    if length > LIMIT || count > MAX_RUNS {
        return Err(invalid());
    }
    let mut restored = Vec::with_capacity(length);
    let mut literals = literals;
    for _ in 0..count {
        let left = number(input)?;
        let base = number(input)?;
        let start = number(input)?;
        let copies = number(input)?;
        if copies < MIN_COPY || left.saturating_add(copies) > length - restored.len() {
            return Err(invalid());
        }
        restored.extend_from_slice(take(&mut literals, left)?);
        let data = prior.get(base).ok_or_else(invalid)?;
        restored.extend_from_slice(
            data.get(start..start.saturating_add(copies))
                .ok_or_else(invalid)?,
        );
    }
    if literals.len() != length - restored.len() {
        return Err(invalid());
    }
    restored.extend_from_slice(literals);
    Ok(restored)
}

fn restore(stage: &Path, output: &Path) -> io::Result<()> {
    let payload = stage.join("payload");
    let data = read_bounded(&stage.join("recipes"), LIMIT)?;
    let mut input = data.as_slice();
    let n = number(&mut input)?;
    if n > 4096 {
        return Err(invalid());
    }
    let (dirs, files) = inventory(&payload)?;
    if files.len() != n {
        return Err(invalid());
    }
    let mut count = number(&mut input)?;
    let mut next = if count == 0 { n } else { number(&mut input)? };
    fs::create_dir(output)?;
    for dir in dirs {
        fs::create_dir_all(output.join(dir))?;
    }
    let mut total = 0;
    let mut prior = Vec::new();
    for (id, name) in files.iter().enumerate() {
        let literal = read_bounded(&payload.join(name), LIMIT - total)?;
        let restored = if id == next {
            let bytes = rebuild(&mut input, &literal, &prior)?;
            count -= 1;
            next = if count == 0 { n } else { number(&mut input)? };
            bytes
        } else {
            literal
        };
        total += restored.len();
        if total > LIMIT {
            return Err(invalid());
        }
        fs::write(output.join(name), &restored)?;
        prior.push(restored);
    }
    if !input.is_empty() || count != 0 {
        return Err(invalid());
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let arguments: Vec<_> = env::args_os().collect();
    let source = Path::new(arguments.get(2).ok_or_else(invalid)?);
    let target = Path::new(arguments.get(3).ok_or_else(invalid)?);
    match arguments.get(1).and_then(|arg| arg.to_str()) {
        Some("prepare") => prepare(source, target),
        Some("restore") => restore(source, target),
        _ => Err(invalid()),
    }
}
