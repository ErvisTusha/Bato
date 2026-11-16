// Original lossless fragment pilot using the LZ77 distance/length principle.
// Source ranges name an earlier native archive member, regardless of segment.
use std::{
    collections::HashMap,
    env, fs,
    io::{self, ErrorKind, Read, Write},
    path::{Component, Path, PathBuf},
};

const LIMIT: usize = 8 * 1024 * 1024;
const MAX_WINDOW: usize = 1 << 30;
const MAX_MEMBERS: usize = 4096;
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

struct Factored {
    literals: Vec<Option<Vec<u8>>>,
    recipes: Vec<u8>,
    count: usize,
}

fn factor_members(raw: &[Vec<u8>], limit: usize) -> Factored {
    let mut literals = Vec::with_capacity(raw.len());
    let mut index = HashMap::<u64, [Option<Spot>; 2]>::new();
    let mut recipes = Vec::new();
    let mut count = 0;
    for (id, member) in raw.iter().enumerate() {
        let previous = &raw[..id];
        let duplicate = previous.iter().any(|other| other == member);
        let (residual, commands) = if duplicate {
            (Vec::new(), Vec::new())
        } else {
            factor(member, previous, &index)
        };
        let use_recipe = residual.len() + 16 * commands.len() + 24 < member.len() * 3 / 4
            && !commands.is_empty();
        if use_recipe {
            count += 1;
            push(&mut recipes, id);
            push(&mut recipes, member.len());
            push(&mut recipes, commands.len());
            for command in commands {
                for value in command {
                    push(&mut recipes, value as usize);
                }
            }
            literals.push(Some(residual));
        } else {
            literals.push(None);
        }
        if !duplicate {
            for offset in 0..member.len().saturating_sub(MIN_COPY - 1) {
                if let Some(key) = anchor(member, offset) {
                    let spot = Some((id, offset));
                    if let Some(spots) = index.get_mut(&key) {
                        *spots = [spot, spots[0]];
                    } else if index.len() < limit / 16 {
                        index.insert(key, [spot, None]);
                    }
                }
            }
        }
    }
    Factored {
        literals,
        recipes,
        count,
    }
}

fn write_group(
    root: &Path,
    (dirs, files): (&[PathBuf], &[PathBuf]),
    raw: &[Vec<u8>],
    factored: Factored,
) -> io::Result<()> {
    let payload = root.join("payload");
    fs::create_dir_all(&payload)?;
    for dir in dirs {
        fs::create_dir_all(payload.join(dir))?;
    }
    for (id, path) in files.iter().enumerate() {
        relative(path)?;
        fs::write(
            payload.join(path),
            factored.literals[id].as_deref().unwrap_or(&raw[id]),
        )?;
    }
    let mut manifest = Vec::new();
    push(&mut manifest, files.len());
    push(&mut manifest, factored.count);
    manifest.extend(factored.recipes);
    fs::write(root.join("recipes"), manifest)
}

fn prepare(root: &Path, stage: &Path) -> io::Result<()> {
    let (dirs, files) = inventory(root)?;
    if files.len() > MAX_MEMBERS {
        return Ok(());
    }
    let mut remaining = LIMIT;
    for path in &files {
        let length =
            usize::try_from(fs::metadata(root.join(path))?.len()).map_err(|_| invalid())?;
        let Some(next) = remaining.checked_sub(length) else {
            return Ok(());
        };
        remaining = next;
    }
    remaining = LIMIT;
    let mut raw = Vec::with_capacity(files.len());
    for path in &files {
        let bytes = read_bounded(&root.join(path), remaining)?;
        remaining -= bytes.len();
        raw.push(bytes);
    }
    let factored = factor_members(&raw, LIMIT);
    if factored.count == 0 {
        return Ok(());
    }
    write_group(stage, (&dirs, &files), &raw, factored)
}

fn prepare_file(source: &Path, stage: &Path, piece: usize, window: usize) -> io::Result<bool> {
    if piece == 0 || piece > window || window > MAX_WINDOW {
        return Err(invalid());
    }
    let mut source = io::BufReader::new(fs::File::open(source)?);
    let mut changed = false;
    for group in 0..u64::MAX {
        let mut raw = Vec::new();
        let mut remaining = window;
        while remaining != 0 && raw.len() < MAX_MEMBERS {
            let mut bytes = Vec::new();
            (&mut source)
                .take(remaining.min(piece) as u64)
                .read_to_end(&mut bytes)?;
            if bytes.is_empty() {
                break;
            }
            remaining -= bytes.len();
            raw.push(bytes);
        }
        if raw.is_empty() {
            return Ok(changed);
        }
        let names: Vec<_> = (0..raw.len())
            .map(|id| PathBuf::from(format!("{id:08}")))
            .collect();
        let factored = factor_members(&raw, window);
        changed |= factored.count != 0;
        let path = stage.join(format!("{group:012}"));
        write_group(&path, (&[], &names), &raw, factored)?;
    }
    Err(invalid())
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

fn rebuild(
    input: &mut &[u8],
    literals: &[u8],
    prior: &[Vec<u8>],
    limit: usize,
) -> io::Result<Vec<u8>> {
    let length = number(input)?;
    let count = number(input)?;
    if length > limit || count > MAX_RUNS {
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

fn restore_group(
    stage: &Path,
    limit: usize,
    directories: impl FnOnce(&[PathBuf]) -> io::Result<()>,
    mut emit: impl FnMut(&Path, &[u8]) -> io::Result<()>,
) -> io::Result<()> {
    let payload = stage.join("payload");
    let data = read_bounded(&stage.join("recipes"), limit)?;
    let mut input = data.as_slice();
    let n = number(&mut input)?;
    if n > MAX_MEMBERS {
        return Err(invalid());
    }
    let (dirs, files) = inventory(&payload)?;
    if files.len() != n {
        return Err(invalid());
    }
    let mut count = number(&mut input)?;
    if count > n {
        return Err(invalid());
    }
    let mut next = if count == 0 { n } else { number(&mut input)? };
    directories(&dirs)?;
    let mut remaining = limit;
    let mut prior = Vec::new();
    for (id, name) in files.iter().enumerate() {
        let literal = read_bounded(&payload.join(name), remaining)?;
        let restored = if id == next {
            let bytes = rebuild(&mut input, &literal, &prior, remaining)?;
            count -= 1;
            next = if count == 0 { n } else { number(&mut input)? };
            bytes
        } else {
            literal
        };
        remaining -= restored.len();
        emit(name, &restored)?;
        prior.push(restored);
    }
    if !input.is_empty() || count != 0 {
        return Err(invalid());
    }
    Ok(())
}

fn restore(stage: &Path, output: &Path) -> io::Result<()> {
    restore_group(
        stage,
        LIMIT,
        |dirs| {
            fs::create_dir(output)?;
            for dir in dirs {
                fs::create_dir_all(output.join(dir))?;
            }
            Ok(())
        },
        |name, bytes| fs::write(output.join(name), bytes),
    )
}

fn restore_file(stage: &Path, output: &Path, window: usize) -> io::Result<()> {
    if !(1..=MAX_WINDOW).contains(&window) {
        return Err(invalid());
    }
    let count = fs::read_dir(stage)?.try_fold(0_u64, |count, entry| {
        if entry?.file_type()?.is_dir() {
            count.checked_add(1).ok_or_else(invalid)
        } else {
            Err(invalid())
        }
    })?;
    let mut output = io::BufWriter::new(fs::File::create_new(output)?);
    for number in 0..count {
        let group = stage.join(format!("{number:012}"));
        if !fs::symlink_metadata(&group)?.file_type().is_dir() {
            return Err(invalid());
        }
        let mut piece = 0;
        restore_group(
            &group,
            window,
            |dirs| {
                if dirs.is_empty() {
                    Ok(())
                } else {
                    Err(invalid())
                }
            },
            |name, bytes| {
                if name != Path::new(&format!("{piece:08}")) {
                    return Err(invalid());
                }
                piece += 1;
                output.write_all(bytes)
            },
        )?;
    }
    output.flush()
}

fn main() -> io::Result<()> {
    let arguments: Vec<_> = env::args_os().collect();
    let source = Path::new(arguments.get(2).ok_or_else(invalid)?);
    let target = Path::new(arguments.get(3).ok_or_else(invalid)?);
    let size = |index| {
        arguments
            .get(index)
            .and_then(|arg: &std::ffi::OsString| arg.to_str())
            .and_then(|arg| arg.parse::<usize>().ok())
            .ok_or_else(invalid)
    };
    match arguments.get(1).and_then(|arg| arg.to_str()) {
        Some("prepare") => prepare(source, target),
        Some("restore") => restore(source, target),
        Some("prepare-file") => {
            let changed = prepare_file(source, target, size(4)?, size(5)?)?;
            println!("{}", u8::from(changed));
            Ok(())
        }
        Some("restore-file") => restore_file(source, target, size(4)?),
        _ => Err(invalid()),
    }
}
