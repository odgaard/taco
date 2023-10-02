#ifndef TACO_UTIL_BENCHMARK_H
#define TACO_UTIL_BENCHMARK_H

#include <chrono>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <cmath>

#include <fstream>

#include "taco/error.h"

using namespace std;

namespace taco {
namespace util {

struct TimeResults {
  double mean;
  double stdev;
  double median;
  std::vector<double> times;
  std::vector<double> energy_consumptions;
  int size;

  friend std::ostream& operator<<(std::ostream& os, const TimeResults& t) {
    if (t.size == 1) {
      return os << t.mean;
    }
    else {
      return os << "  mean:   " << t.mean   << endl
                << "  stdev:  " << t.stdev  << endl
                << "  median: " << t.median;
    }
  }
};



typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

/// Monotonic timer that can be called multiple times and that computes
/// statistics such as mean and median from the calls.
class Timer {
public:

  ~Timer() {
    if(dummyA){ free(dummyA); }
    if(dummyB){ free(dummyB); }
  }
  void start() {
    energy_begin = get_energy_consumed(rapl_dir);
    begin = std::chrono::steady_clock::now();
  }

  void stop() {
    auto end = std::chrono::steady_clock::now();
    auto energy_end = get_energy_consumed(rapl_dir);

    auto diff = std::chrono::duration<double, std::milli>(end - begin).count();
    times.push_back(diff);
    
    auto energy_diff = energy_end - energy_begin;
    energy_consumptions.push_back(energy_diff);
  }

  // Function to read energy consumed
  double get_energy_consumed(const std::string& rapl_dir) {
      std::string energy_file_path = rapl_dir + "/energy_uj";
      std::ifstream energy_file(energy_file_path);
      if (!energy_file.is_open()) {
          throw std::runtime_error("Failed to open file: " + energy_file_path);
      }
      long long energy_uj;
      energy_file >> energy_uj;
      if (!energy_file.good()) {
          throw std::runtime_error("Failed to read from file: " + energy_file_path);
      }
      return static_cast<double>(energy_uj) * 1e-6;  // Convert micro-joules to joules
  }

  double med(vector<double> vec) {
      typedef vector<int>::size_type vec_sz;

      vec_sz size = vec.size();
      if (size == 0)
          throw domain_error("median of an empty vector");

      sort(vec.begin(), vec.end());

      vec_sz mid = size/2;

      return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
  }

  // Compute mean, standard deviation and median
  TimeResults getResult() {
    int repeat = static_cast<int>(times.size());

    TimeResults result;
    result.times = times;
    result.energy_consumptions = energy_consumptions;
  
    // times = ends - begins
    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());

    // remove 10% worst and best cases
    const int truncate = static_cast<int>(repeat * 0.1);
    double mean = std::accumulate(sorted_times.begin() + truncate,
                                  sorted_times.end() - truncate, 0.0);
    int size = repeat - 2 * truncate;
    result.size = size;
    mean /= size;
    result.mean = mean;

    std::vector<double> diff(size);
    std::transform(sorted_times.begin() + truncate, sorted_times.end() - truncate,
                   diff.begin(), [mean](double x) { return x - mean; });
    double sq_sum = std::inner_product(diff.begin(), diff.end(),
                                       diff.begin(), 0.0);
    result.stdev = std::sqrt(sq_sum / size);
    result.median = (size % 2)
                    ? sorted_times[size/2]
                    : (sorted_times[size/2-1] + sorted_times[size/2]) / 2;
    return result;
  }

  double clear_cache() {
    double ret = 0.0;
    if (!dummyA) {
      dummyA = (double*)(malloc(dummySize*sizeof(double)));
      dummyB = (double*)(malloc(dummySize*sizeof(double)));
    }
    for (int i=0; i< 100; i++) {
      dummyA[rand() % dummySize] = rand()/RAND_MAX;
      dummyB[rand() % dummySize] = rand()/RAND_MAX;
    }
    for (int i=0; i<dummySize; i++) {
      ret += dummyA[i] * dummyB[i];
    }
    return ret;
  }

protected:
  vector<double> times;
  vector<double> energy_consumptions;
  TimePoint begin;
  double energy_begin;
private:
  std::string rapl_dir = "/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0";
  int dummySize = 3000000;
  double* dummyA = NULL;
  double* dummyB = NULL;
};


/// Monotonic timer that prints results as it goes.
class LapTimer {
public:
  LapTimer(string timerName = "") : timerGroup(true), isTiming(false) {
    if (timerName != "") {
      std::cout << timerName << std::endl;
    }
  }

  void start(const string& name) {
    this->timingName = name;
    taco_iassert(!isTiming) << "Called PrintTimer::start twice in a row";
    isTiming = true;
    begin = std::chrono::steady_clock::now();
  }

  void lap(const string& name) {
    auto end = std::chrono::steady_clock::now();
    taco_iassert(isTiming) << "lap timer that hasn't been started";
    if (timerGroup) {
      std::cout << "  ";
    }
    auto diff = std::chrono::duration<double, std::milli>(end - begin).count();
    std::cout << timingName << ": " << diff << " ms" << std::endl;

    this->timingName = name;
    begin = std::chrono::steady_clock::now();
  }

  void stop() {
    auto end = std::chrono::steady_clock::now();
    taco_iassert(isTiming)
        << "Called PrintTimer::stop without first calling start";
    if (timerGroup) {
      std::cout << "  ";
    }
    auto diff = std::chrono::duration<double, std::milli>(end - begin).count();
    std::cout << timingName << ": " << diff << " ms" << std::endl;
    isTiming = false;
  }

private:
  bool timerGroup;
  string timingName;
  TimePoint begin;
  bool isTiming;
};

}}

#define TACO_TIME_REPEAT(CODE, REPEAT, RES, COLD) {  \
    taco::util::Timer timer;                         \
    for(int i=0; i<REPEAT; i++) {                    \
      if(COLD)                                       \
        timer.clear_cache();                         \
      timer.start();                                 \
      CODE;                                          \
      timer.stop();                                  \
    }                                                \
    RES = timer.getResult();                         \
  }

#endif
