//! CPU-only archive accelerator facade.
//! `--no-default-features` builds Bato without CUDA or wgpu dependencies.
//! Provenance: Bato's existing deterministic CPU fallback.

pub(crate) mod filters {
    use crate::core::types::{BatoError, Compute, ExecutionStats, TransformId};

    /// No device discovery or memory allocation in CPU-only builds.
    #[derive(Default)]
    pub(crate) struct FilterPool {
        _private: (),
    }

    #[derive(Clone, Copy)]
    pub(crate) enum Direction {
        Forward,
        Inverse,
    }

    /// Archive-worker API shared with the optional GPU scheduler.
    pub(crate) struct WorkerFilters<'a> {
        compute: Compute,
        _pool: &'a FilterPool,
        _stats: &'a ExecutionStats,
    }

    impl<'a> WorkerFilters<'a> {
        pub(crate) fn new(
            pool: &'a FilterPool,
            (compute, stats): (Compute, &'a ExecutionStats),
            _worker: usize,
        ) -> Self {
            Self {
                compute,
                _pool: pool,
                _stats: stats,
            }
        }

        pub(crate) fn run(
            &mut self,
            _input: &[u8],
            _transform: TransformId,
            _direction: Direction,
        ) -> Result<Option<Vec<u8>>, BatoError> {
            match self.compute {
                Compute::Cpu | Compute::Auto => Ok(None),
                Compute::Cuda => Err(BatoError::Cuda(
                    "Bato was built without GPU support; rebuild with --features gpu".into(),
                )),
                Compute::Wgpu => Err(BatoError::Wgpu(
                    "Bato was built without GPU support; rebuild with --features gpu".into(),
                )),
                Compute::Hip | Compute::LevelZero | Compute::Metal => Err(BatoError::Native(
                    "Bato was built without native GPU support".into(),
                )),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn auto_and_cpu_use_the_existing_cpu_filters() {
            let pool = FilterPool::default();
            let stats = ExecutionStats::default();
            for compute in [Compute::Auto, Compute::Cpu] {
                let mut worker = WorkerFilters::new(&pool, (compute, &stats), 0);
                assert!(worker
                    .run(&[1, 2, 3], TransformId::Delta1, Direction::Forward)
                    .unwrap()
                    .is_none());
            }
        }

        #[test]
        fn explicit_gpu_requests_are_rejected_clearly() {
            let pool = FilterPool::default();
            let stats = ExecutionStats::default();
            for compute in [
                Compute::Cuda,
                Compute::Wgpu,
                Compute::Hip,
                Compute::LevelZero,
                Compute::Metal,
            ] {
                let mut worker = WorkerFilters::new(&pool, (compute, &stats), 0);
                assert!(worker
                    .run(&[1, 2, 3], TransformId::Delta1, Direction::Forward)
                    .is_err());
            }
        }
    }
}
