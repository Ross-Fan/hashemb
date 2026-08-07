#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace hashemb {

/// Portable barrier using std::mutex + std::condition_variable.
class Barrier {
 public:
  Barrier() : threshold_(0), count_(0), gen_(0) {}

  explicit Barrier(int count) : threshold_(count), count_(count), gen_(0) {}

  void init(int count) {
    threshold_ = count;
    count_ = count;
    gen_ = 0;
  }

  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    int local_gen = gen_;
    if (--count_ == 0) {
      count_ = threshold_;
      ++gen_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this, local_gen] { return gen_ != local_gen; });
    }
  }

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  int threshold_;
  int count_;
  int gen_;
};

/// Persistent thread pool using portable Barrier for synchronization.
/// Workers are created once (lazy singleton) and reused across all calls.
class ThreadPool {
 public:
  static ThreadPool& instance() {
    static ThreadPool pool;
    return pool;
  }

  /// Run fn(i) for i in [0, n) in parallel (master participates too).
  /// Falls back to sequential when work is too small.
  template <typename Func>
  void parallel_for(size_t n, Func&& fn) {
    // Skip threading when there aren't enough items for each worker.
    if (nworkers_ == 0 || n < static_cast<size_t>(nworkers_ + 1)) {
      for (size_t i = 0; i < n; ++i) fn(i);
      return;
    }

    work_fn_ = [&fn](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) fn(i);
    };
    total_ = n;

    // Phase 1: release workers (master + nworkers_)
    barrier_.arrive_and_wait();

    // Master does its share
    do_chunk(nworkers_);  // master = last "worker"

    // Phase 2: wait for all workers to finish
    barrier_.arrive_and_wait();
  }

 private:
  ThreadPool() {
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    nworkers_ = hw > 1 ? hw - 1 : 0;  // leave 1 core for master
    if (nworkers_ > 0) {
      barrier_.init(nworkers_ + 1);
      for (int i = 0; i < nworkers_; ++i) {
        workers_.emplace_back(&ThreadPool::worker_loop, this, i);
      }
    }
  }

  ~ThreadPool() {
    if (nworkers_ == 0) return;
    stop_.store(true, std::memory_order_release);
    barrier_.arrive_and_wait();   // wake workers from Phase 1
    barrier_.arrive_and_wait();   // let them exit via Phase 2
    for (auto& w : workers_) w.join();
  }

  void do_chunk(int tid) {
    size_t chunk = total_ / (nworkers_ + 1);
    size_t rem = total_ % (nworkers_ + 1);
    size_t start = static_cast<size_t>(tid) * chunk + std::min<size_t>(static_cast<size_t>(tid), rem);
    size_t end = start + chunk + (static_cast<size_t>(tid) < rem ? 1 : 0);
    if (start < end) work_fn_(start, end);
  }

  void worker_loop(int tid) {
    while (true) {
      // Phase 1: wait for work
      barrier_.arrive_and_wait();

      if (stop_.load(std::memory_order_acquire)) {
        barrier_.arrive_and_wait();  // Phase 2: keep barrier balanced
        break;
      }

      do_chunk(tid);

      // Phase 2: signal done
      barrier_.arrive_and_wait();
    }
  }

  int nworkers_ = 0;
  std::vector<std::thread> workers_;
  Barrier barrier_;
  std::atomic<bool> stop_{false};

  // Work descriptor (set by master before releasing workers)
  std::function<void(size_t, size_t)> work_fn_;
  size_t total_ = 0;
};

}  // namespace hashemb
