//! Platform probes for the memory a process may still allocate:
//! available RAM, cgroup limits and address-space limits.
use super::super::types::BatoError;

#[cfg(not(windows))]
fn probe_error(message: impl Into<String>) -> BatoError {
    BatoError::InvalidOptions(format!(
        "cannot resolve host memory allowance: {}",
        message.into()
    ))
}
fn to_usize(bytes: u64) -> usize {
    usize::try_from(bytes).unwrap_or(usize::MAX)
}

#[cfg(target_os = "linux")]
pub(super) fn effective_available_memory() -> Result<usize, BatoError> {
    use std::fs;
    let meminfo = fs::read_to_string("/proc/meminfo")?;
    let available = meminfo
        .lines()
        .find_map(|line| {
            line.strip_prefix("MemAvailable:")
                .and_then(|n| n.split_whitespace().next())
                .and_then(|n| n.parse::<u64>().ok())
        })
        .and_then(|n| n.checked_mul(1024))
        .ok_or_else(|| probe_error("missing MemAvailable"))?;
    let membership = fs::read_to_string("/proc/self/cgroup")?;
    let mounts = fs::read_to_string("/proc/self/mountinfo")?;
    let mut available = cgroup_available(available, &cgroup_locations(&membership, &mounts)?)?;
    let pages = fs::read_to_string("/proc/self/statm")?
        .split_whitespace()
        .next()
        .and_then(|s| s.parse::<u64>().ok())
        .ok_or_else(|| probe_error("invalid process virtual memory"))?;
    // SAFETY: sysconf takes a constant selector and writes no caller memory.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let virtual_bytes = pages
        .checked_mul(u64::try_from(page_size).map_err(|_| probe_error("invalid page size"))?)
        .ok_or_else(|| probe_error("virtual-memory overflow"))?;
    let status = fs::read_to_string("/proc/self/status")?;
    let data_bytes = status
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmData:")
                .and_then(|n| n.split_whitespace().next())
                .and_then(|n| n.parse::<u64>().ok())
        })
        .and_then(|n| n.checked_mul(1024))
        .ok_or_else(|| probe_error("missing process data size"))?;
    available = available
        .min(rlimit_remaining(libc::RLIMIT_AS as _, virtual_bytes)?)
        .min(rlimit_remaining(libc::RLIMIT_DATA as _, data_bytes)?);
    Ok(to_usize(available))
}

/// The cgroup memory-controller interface a hierarchy exposes.
#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cgroup {
    V1,
    V2,
}
#[cfg(target_os = "linux")]
impl Cgroup {
    /// Files holding the hierarchy's memory caps.
    fn caps(self) -> &'static [&'static str] {
        match self {
            Self::V1 => &["memory.limit_in_bytes"],
            Self::V2 => &["memory.max", "memory.high"],
        }
    }
    /// File holding the hierarchy's current memory use.
    fn usage(self) -> &'static str {
        match self {
            Self::V1 => "memory.usage_in_bytes",
            Self::V2 => "memory.current",
        }
    }
    /// `memory.stat` keys for dirty and writeback file pages.
    fn pending_keys(self) -> [&'static str; 2] {
        match self {
            Self::V1 => ["dirty", "writeback"],
            Self::V2 => ["file_dirty", "file_writeback"],
        }
    }
}

#[cfg(target_os = "linux")]
fn cgroup_file_cache_credit(stats: &str, version: Cgroup) -> Result<u64, BatoError> {
    // Use file LRUs, not anon/shmem/unevictable counters. Some kernels also
    // put reclaimable MADV_FREE pages on the file LRU. Keep half of the clean
    // cache, and give no credit for dirty or writeback pages.
    // This is a conservative estimate, not a kernel MemAvailable counter.
    let [dirty_key, writeback_key] = version.pending_keys();
    let keys = ["active_file", "inactive_file", dirty_key, writeback_key];
    let mut values = [None; 4];
    for line in stats.lines() {
        let mut fields = line.split_whitespace();
        let Some(key) = fields.next() else { continue };
        let Some(index) = keys.iter().position(|&name| name == key) else {
            continue;
        };
        let value = fields
            .next()
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or_else(|| probe_error("invalid cgroup file-cache statistic"))?;
        if fields.next().is_some() || values[index].replace(value).is_some() {
            return Err(probe_error(
                "duplicate or invalid cgroup file-cache statistic",
            ));
        }
    }
    let [Some(active), Some(inactive), Some(dirty), Some(writeback)] = values else {
        return Err(probe_error("missing cgroup file-cache statistic"));
    };
    let file = active
        .checked_add(inactive)
        .ok_or_else(|| probe_error("cgroup file-cache overflow"))?;
    let pending = dirty
        .checked_add(writeback)
        .ok_or_else(|| probe_error("cgroup dirty-cache overflow"))?;
    Ok(file.saturating_sub(pending) / 2)
}

#[cfg(target_os = "linux")]
fn cgroup_available(
    mut available: u64,
    locations: &[(std::path::PathBuf, std::path::PathBuf, Cgroup)],
) -> Result<u64, BatoError> {
    for (mount, leaf, version) in locations {
        let mut credit = CacheCredit {
            leaf,
            version: *version,
            value: None,
        };
        let mut current = leaf.as_path();
        loop {
            available = available.min(cgroup_room(current, &mut credit)?);
            if current == mount.as_path() {
                break;
            }
            current = current
                .parent()
                .filter(|p| p.starts_with(mount))
                .ok_or_else(|| probe_error("cgroup escaped its mount"))?;
        }
    }
    Ok(available)
}

/// The leaf cgroup's reclaimable file cache, read once on first use.
#[cfg(target_os = "linux")]
struct CacheCredit<'a> {
    leaf: &'a std::path::Path,
    version: Cgroup,
    value: Option<u64>,
}
#[cfg(target_os = "linux")]
impl CacheCredit<'_> {
    /// Returns the credit, reading `memory.stat` on the first call.
    fn get_or_read(&mut self) -> Result<u64, BatoError> {
        if let Some(value) = self.value {
            return Ok(value);
        }
        // Only this cgroup's cache can relieve each ancestor limit. Never
        // count an ancestor's sibling cache. V1 deliberately uses local
        // fields, not total_* fields.
        let value = match std::fs::read_to_string(self.leaf.join("memory.stat")) {
            Ok(stats) => cgroup_file_cache_credit(&stats, self.version)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
            Err(e) => return Err(e.into()),
        };
        self.value = Some(value);
        Ok(value)
    }
}

/// Room left under every memory cap set on one cgroup directory.
#[cfg(target_os = "linux")]
fn cgroup_room(dir: &std::path::Path, credit: &mut CacheCredit) -> Result<u64, BatoError> {
    let mut room = u64::MAX;
    for cap in credit.version.caps() {
        let Some(cap) = cgroup_cap(&dir.join(cap))? else {
            continue;
        };
        let usage = std::fs::read_to_string(dir.join(credit.version.usage()))?
            .trim()
            .parse::<u64>()
            .map_err(|_| probe_error("invalid cgroup usage"))?;
        let working_set = usage.saturating_sub(credit.get_or_read()?.min(usage));
        room = room.min(cap.saturating_sub(working_set));
    }
    Ok(room)
}

/// A cgroup memory cap, or `None` when the file is missing or says "max".
#[cfg(target_os = "linux")]
fn cgroup_cap(path: &std::path::Path) -> Result<Option<u64>, BatoError> {
    let value = match std::fs::read_to_string(path) {
        Ok(value) => value,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    if value.trim() == "max" {
        return Ok(None);
    }
    value
        .trim()
        .parse::<u64>()
        .map(Some)
        .map_err(|_| probe_error("invalid cgroup limit"))
}

#[cfg(target_os = "linux")]
fn cgroup_locations(
    membership: &str,
    mounts: &str,
) -> Result<Vec<(std::path::PathBuf, std::path::PathBuf, Cgroup)>, BatoError> {
    use std::path::{Component, Path, PathBuf};
    fn unescape(s: &str) -> Result<PathBuf, BatoError> {
        let mut bytes = Vec::new();
        let input = s.as_bytes();
        let mut i = 0;
        while i < input.len() {
            if input[i] == b'\\' && i + 3 < input.len() {
                let octal = std::str::from_utf8(&input[i + 1..i + 4])
                    .ok()
                    .and_then(|s| u8::from_str_radix(s, 8).ok())
                    .ok_or_else(|| probe_error("invalid mount escape"))?;
                bytes.push(octal);
                i += 4;
            } else {
                bytes.push(input[i]);
                i += 1;
            }
        }
        use std::os::unix::ffi::OsStringExt;
        Ok(PathBuf::from(std::ffi::OsString::from_vec(bytes)))
    }
    let mut result = Vec::new();
    for line in membership.lines() {
        let parts: Vec<_> = line.splitn(3, ':').collect();
        if parts.len() != 3 {
            return Err(probe_error("invalid cgroup membership"));
        }
        let version = if parts[0] == "0" && parts[1].is_empty() {
            Cgroup::V2
        } else {
            Cgroup::V1
        };
        if version == Cgroup::V1 && !parts[1].split(',').any(|s| s == "memory") {
            continue;
        }
        let name = Path::new(parts[2]);
        if !name.is_absolute() || name.components().any(|c| matches!(c, Component::ParentDir)) {
            return Err(probe_error("unsafe cgroup membership"));
        }
        for mount_line in mounts.lines() {
            let Some((before, after)) = mount_line.split_once(" - ") else {
                continue;
            };
            let fields: Vec<_> = before.split_whitespace().collect();
            let fs_fields: Vec<_> = after.split_whitespace().collect();
            if fields.len() < 5 || fs_fields.len() < 3 {
                continue;
            }
            let matches = match version {
                Cgroup::V2 => fs_fields[0] == "cgroup2",
                Cgroup::V1 => {
                    fs_fields[0] == "cgroup" && fs_fields[2].split(',').any(|s| s == "memory")
                }
            };
            if !matches {
                continue;
            }
            let root = unescape(fields[3])?;
            let mount = unescape(fields[4])?;
            let relative = if let Ok(relative) = name.strip_prefix(&root) {
                relative
            }
            // A namespaced cgroup path is relative to the exposed root.
            else if name == Path::new("/") {
                Path::new("")
            } else {
                continue;
            };
            let leaf = mount.join(relative);
            if !leaf.starts_with(&mount) {
                return Err(probe_error("cgroup escaped its mount"));
            }
            if !result.iter().any(|(m, l, _)| m == &mount && l == &leaf) {
                result.push((mount, leaf, version));
            }
        }
    }
    // Linux without an enabled memory controller has no cgroup memory cap.
    Ok(result)
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn rlimit_remaining(resource: libc::c_int, used: u64) -> Result<u64, BatoError> {
    let mut limits = std::mem::MaybeUninit::<libc::rlimit>::uninit();
    // SAFETY: valid rlimit output pointer and OS resource selector.
    if unsafe { libc::getrlimit(resource as _, limits.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: getrlimit succeeded, so it initialized `limits`.
    let limits = unsafe { limits.assume_init() };
    Ok(if limits.rlim_cur == libc::RLIM_INFINITY {
        u64::MAX
    } else {
        // rlim_t is narrower than u64 on some targets, so the cast is not
        // redundant everywhere.
        #[allow(clippy::unnecessary_cast)]
        let limit = limits.rlim_cur as u64;
        limit.saturating_sub(used)
    })
}

#[cfg(windows)]
pub(super) fn effective_available_memory() -> Result<usize, BatoError> {
    use windows_sys::Win32::System::{
        SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX},
        Threading::{
            GetCurrentProcess, GetProcessInformation, ProcessAppMemoryInfo, APP_MEMORY_INFORMATION,
        },
    };
    // SAFETY: MEMORYSTATUSEX is a plain C struct; all-zero bytes are valid.
    let mut physical: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    physical.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: both structures have the exact SDK layout and declared byte lengths.
    if unsafe { GlobalMemoryStatusEx(&mut physical) } == 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: APP_MEMORY_INFORMATION is a plain C struct; all-zero bytes are valid.
    let mut process: APP_MEMORY_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: the output pointer and byte length describe `process` exactly.
    if unsafe {
        GetProcessInformation(
            GetCurrentProcess(),
            ProcessAppMemoryInfo,
            (&mut process as *mut APP_MEMORY_INFORMATION).cast(),
            std::mem::size_of::<APP_MEMORY_INFORMATION>() as u32,
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error().into());
    }
    // ProcessAppMemoryInfo reports additional commit available to this process,
    // including effective app/job limits, rather than only machine-wide RAM.
    Ok(to_usize(
        physical
            .ullAvailPhys
            .min(physical.ullAvailVirtual)
            .min(process.AvailableCommit),
    ))
}

#[cfg(target_os = "macos")]
#[allow(deprecated)] // Stable Mach ABI; libc deprecates its bindings, not these OS calls.
pub(super) fn effective_available_memory() -> Result<usize, BatoError> {
    unsafe extern "C" {
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }
    // SAFETY: vm_statistics64 is a plain C struct; all-zero bytes are valid.
    let mut vm: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: the Mach output pointer, flavor and count match vm_statistics64.
    let host = unsafe { libc::mach_host_self() };
    // SAFETY: as above; `host` is the port just obtained.
    let status = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            (&mut vm as *mut libc::vm_statistics64).cast(),
            &mut count,
        )
    };
    // SAFETY: releases the send right obtained by mach_host_self above,
    // exactly once.
    unsafe {
        mach_port_deallocate(libc::mach_task_self(), host);
    }
    if status != libc::KERN_SUCCESS {
        return Err(probe_error("host_statistics64 failed"));
    }
    // SAFETY: sysconf takes no pointers and only reads a configuration value.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let mut available = (vm.free_count as u64)
        .checked_add(vm.inactive_count as u64)
        .and_then(|n| n.checked_mul(u64::try_from(page_size).ok()?))
        .ok_or_else(|| probe_error("invalid available page count"))?;
    // SAFETY: mach_task_basic_info is a plain C struct; all-zero bytes are valid.
    let mut task: libc::mach_task_basic_info = unsafe { std::mem::zeroed() };
    let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
    // SAFETY: the output pointer, flavor and count match mach_task_basic_info.
    if unsafe {
        libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            (&mut task as *mut libc::mach_task_basic_info).cast(),
            &mut count,
        )
    } != libc::KERN_SUCCESS
    {
        return Err(probe_error("task_info failed"));
    }
    available = available
        .min(rlimit_remaining(libc::RLIMIT_AS, task.virtual_size)?)
        .min(rlimit_remaining(libc::RLIMIT_DATA, task.virtual_size)?);
    Ok(to_usize(available))
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
pub(super) fn effective_available_memory() -> Result<usize, BatoError> {
    Err(probe_error("unsupported host platform"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "linux")]
    #[test]
    fn cgroup_mount_roots_nested_paths_and_v1_are_resolved() {
        use std::path::PathBuf;
        let mounts = "1 2 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n2 2 0:2 /docker /delegated\\040memory rw - cgroup memory rw,memory\n";
        let locations =
            cgroup_locations("0::/parent/leaf\n5:memory:/docker/team/job\n", mounts).unwrap();
        assert_eq!(
            locations,
            vec![
                (
                    PathBuf::from("/sys/fs/cgroup"),
                    PathBuf::from("/sys/fs/cgroup/parent/leaf"),
                    Cgroup::V2
                ),
                (
                    PathBuf::from("/delegated memory"),
                    PathBuf::from("/delegated memory/team/job"),
                    Cgroup::V1
                )
            ]
        );
        assert!(cgroup_locations("0::/../../other", mounts).is_err());
        assert_eq!(
            cgroup_locations(
                "0::/",
                "1 2 0:1 /host/container /sys/fs/cgroup rw - cgroup2 cgroup rw"
            )
            .unwrap()[0]
                .1,
            PathBuf::from("/sys/fs/cgroup")
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cgroup_cache_credit_counts_only_clean_file_lrus() {
        let stats = "anon 999999\nshmem 999999\nunevictable 999999\nactive_file 700\ninactive_file 300\nfile_dirty 100\nfile_writeback 200\n";
        assert_eq!(cgroup_file_cache_credit(stats, Cgroup::V2).unwrap(), 350);
        let stats = "total_active_file 999999\ntotal_inactive_file 999999\nactive_file 70\ninactive_file 30\ndirty 10\nwriteback 20\n";
        assert_eq!(cgroup_file_cache_credit(stats, Cgroup::V1).unwrap(), 35);
        assert_eq!(
            cgroup_file_cache_credit(
                "active_file 10\ninactive_file 20\nfile_dirty 30\nfile_writeback 40\n",
                Cgroup::V2
            )
            .unwrap(),
            0
        );
        // memory.stat is keyed, not positional; the kernel can add fields.
        assert_eq!(
            cgroup_file_cache_credit(
                "file_writeback 0\nnew_field 999999\nfile_dirty 0\ninactive_file 0\nactive_file 3\n", Cgroup::V2)
            .unwrap(),
            1
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cgroup_cache_statistics_fail_closed_on_bad_or_missing_fields() {
        let good = "active_file 10\ninactive_file 20\nfile_dirty 1\nfile_writeback 2\n";
        for bad in [
            good.replace("active_file 10\n", ""),
            good.replace("active_file 10", "active_file invalid"),
            good.replace("active_file 10", "active_file -1"),
            good.replace("active_file 10", "active_file 10 extra"),
            format!("{good}active_file 10\n"),
            good.replace("active_file 10", "active_file 18446744073709551615"),
            good.replace("file_dirty 1\n", "file_dirty 18446744073709551615\n"),
        ] {
            assert!(cgroup_file_cache_credit(&bad, Cgroup::V2).is_err(), "{bad}");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cgroup_cache_credit_is_leaf_local_and_preserves_all_caps() {
        use std::fs;
        let root = tempfile::tempdir().unwrap();
        let leaf = root.path().join("child");
        fs::create_dir(&leaf).unwrap();
        fs::write(root.path().join("memory.max"), "1000").unwrap();
        fs::write(root.path().join("memory.high"), "500").unwrap();
        fs::write(root.path().join("memory.current"), "480").unwrap();
        fs::write(leaf.join("memory.max"), "900").unwrap();
        fs::write(leaf.join("memory.current"), "100").unwrap();
        // Even an invalid ancestor statistic must not be read or credited.
        fs::write(root.path().join("memory.stat"), "active_file invalid").unwrap();
        let locations = [(root.path().into(), leaf.clone(), Cgroup::V2)];
        assert_eq!(cgroup_available(2000, &locations).unwrap(), 20);
        fs::write(
            leaf.join("memory.stat"),
            "active_file 40\ninactive_file 20\nfile_dirty 10\nfile_writeback 10\n",
        )
        .unwrap();
        assert_eq!(cgroup_available(2000, &locations).unwrap(), 40);
        assert_eq!(cgroup_available(30, &locations).unwrap(), 30);
        fs::write(root.path().join("memory.current"), "510").unwrap();
        assert_eq!(cgroup_available(2000, &locations).unwrap(), 10);
        fs::write(root.path().join("memory.current"), "530").unwrap();
        assert_eq!(cgroup_available(2000, &locations).unwrap(), 0);
        // Cache credit cannot exceed observed usage or bypass a small leaf cap.
        fs::write(root.path().join("memory.high"), "max").unwrap();
        fs::write(root.path().join("memory.current"), "0").unwrap();
        fs::write(leaf.join("memory.max"), "7").unwrap();
        fs::write(leaf.join("memory.current"), "1").unwrap();
        assert_eq!(cgroup_available(2000, &locations).unwrap(), 7);
        fs::write(root.path().join("memory.limit_in_bytes"), "700").unwrap();
        fs::write(root.path().join("memory.usage_in_bytes"), "640").unwrap();
        fs::write(leaf.join("memory.limit_in_bytes"), "900").unwrap();
        fs::write(leaf.join("memory.usage_in_bytes"), "200").unwrap();
        fs::write(
            leaf.join("memory.stat"),
            "active_file 70\ninactive_file 30\ndirty 10\nwriteback 20\ntotal_active_file 999999\n",
        )
        .unwrap();
        assert_eq!(
            cgroup_available(2000, &[(root.path().into(), leaf, Cgroup::V1)]).unwrap(),
            95
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cgroup_admission_respects_ancestor_usage_and_soft_caps() {
        use std::fs;
        let root = tempfile::tempdir().unwrap();
        let leaf = root.path().join("child");
        fs::create_dir(&leaf).unwrap();
        fs::write(root.path().join("memory.max"), "1000").unwrap();
        fs::write(root.path().join("memory.high"), "800").unwrap();
        fs::write(root.path().join("memory.current"), "600").unwrap();
        fs::write(leaf.join("memory.max"), "900").unwrap();
        fs::write(leaf.join("memory.high"), "max").unwrap();
        fs::write(leaf.join("memory.current"), "100").unwrap();
        let locations = [(root.path().into(), leaf.clone(), Cgroup::V2)];
        assert_eq!(cgroup_available(2000, &locations).unwrap(), 200);
        fs::write(leaf.join("memory.high"), "150").unwrap();
        assert_eq!(cgroup_available(2000, &locations).unwrap(), 50);
        fs::write(leaf.join("memory.current"), "200").unwrap();
        assert_eq!(cgroup_available(2000, &locations).unwrap(), 0);
        fs::write(root.path().join("memory.limit_in_bytes"), "700").unwrap();
        fs::write(root.path().join("memory.usage_in_bytes"), "640").unwrap();
        fs::write(leaf.join("memory.limit_in_bytes"), "900").unwrap();
        fs::write(leaf.join("memory.usage_in_bytes"), "200").unwrap();
        assert_eq!(
            cgroup_available(2000, &[(root.path().into(), leaf, Cgroup::V1)]).unwrap(),
            60
        );
    }
}
