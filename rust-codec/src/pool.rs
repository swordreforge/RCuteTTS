//! Persistent thread pool with scoped fork-join (QORA-gpool lesson).
//!
//! `std::thread::scope` spawns OS threads per call (~15us x N); prefill
//! issues ~70 GEMM calls and decode hundreds of GEMVs — all overhead.
//! Workers here are persistent threads blocked on a shared queue; [`scope`]
//! runs borrowed closures and returns when all finish (same soundness
//! argument as `thread::scope`: the join is a barrier, so lifetime
//! extension of stack closures is safe).
//!
//! Pool size: `CUTETTS_THREADS` override, else logical CPUs.

use std::sync::{Arc, Condvar, Mutex, OnceLock};

type RawFn = *mut (dyn FnOnce() + Send);

/// Raw job pointer crossing into the queue. `Send` is sound: workers only
/// dereference it while [`scope`]'s join barrier guarantees the stack frame
/// is alive (same contract as `std::thread::scope`).
struct JobPtr(RawFn);
unsafe impl Send for JobPtr {}

struct Shared {
    queue: Mutex<Vec<JobPtr>>,
    nonzero: Condvar, // signaled on push
    pending: Mutex<usize>,
    done: Condvar, // signaled when pending hits 0
}

struct Pool {
    shared: Arc<Shared>,
    workers: usize,
}

fn global_pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let avail = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).max(1);
        let workers = std::env::var("CUTETTS_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(avail);
        let shared = Arc::new(Shared {
            queue: Mutex::new(Vec::new()),
            nonzero: Condvar::new(),
            pending: Mutex::new(0),
            done: Condvar::new(),
        });
        for _ in 0..workers {
            let sh = Arc::clone(&shared);
            std::thread::spawn(move || loop {
                let job = {
                    let mut q = sh.queue.lock().unwrap();
                    while q.is_empty() {
                        q = sh.nonzero.wait(q).unwrap();
                    }
                    q.pop().unwrap().0
                };
                unsafe {
                    (Box::from_raw(job) as Box<dyn FnOnce() + Send>)();
                }
                let mut p = sh.pending.lock().unwrap();
                *p -= 1;
                if *p == 0 {
                    // notify_all: independent scope() calls may wait
                    // concurrently on this shared counter; notify_one would
                    // strand all but one waiter.
                    sh.done.notify_all();
                }
            });
        }
        Pool { shared, workers }
    })
}

/// Number of worker threads (for chunking math).
pub fn num_workers() -> usize {
    global_pool().workers
}

/// Run borrowed closures on the pool; return when all complete.
/// `jobs` must be non-empty... (empty is a no-op returning immediately).
pub fn scope<'s>(jobs: Vec<Box<dyn FnOnce() + Send + 's>>) {
    if jobs.is_empty() {
        return;
    }
    let pool = global_pool();
    {
        let mut p = pool.shared.pending.lock().unwrap();
        *p += jobs.len();
    }
    {
        let mut q = pool.shared.queue.lock().unwrap();
        for job in jobs {
            // SAFETY: the barrier below joins all jobs before `scope`
            // returns, so stack-borrowing closures cannot outlive their
            // frames — same contract as `std::thread::scope`.
            let raw = Box::into_raw(job);
            let raw: *mut (dyn FnOnce() + Send + 'static) =
                unsafe { std::mem::transmute(raw) };
            q.push(JobPtr(raw as RawFn));
        }
        pool.shared.nonzero.notify_all();
    }
    let mut p = pool.shared.pending.lock().unwrap();
    while *p > 0 {
        p = pool.shared.done.wait(p).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn echo_and_sum() {
        let mut out = vec![0u64; 64];
        let jobs: Vec<Box<dyn FnOnce() + Send + '_>> = out
            .chunks_mut(8)
            .enumerate()
            .map(|(i, chunk)| {
                Box::new(move || {
                    for (j, v) in chunk.iter_mut().enumerate() {
                        *v = (i * 8 + j) as u64;
                    }
                }) as Box<dyn FnOnce() + Send + '_>
            })
            .collect();
        scope(jobs);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, i as u64);
        }
    }

    #[test]
    fn parallel_counter() {
        let n = AtomicUsize::new(0);
        let jobs: Vec<Box<dyn FnOnce() + Send + '_>> = (0..128)
            .map(|_| {
                Box::new(|| {
                    n.fetch_add(1, Ordering::Relaxed);
                }) as Box<dyn FnOnce() + Send + '_>
            })
            .collect();
        scope(jobs);
        assert_eq!(n.load(Ordering::Relaxed), 128);
    }

    #[test]
    fn empty_is_noop() {
        scope(vec![]);
    }
}
