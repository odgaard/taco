#ifndef TACO_OP_H
#define TACO_OP_H

#include <iostream>
#include "taco.h"
#include <taco/util/timers.h>

#include <fstream>
#include <chrono>
#include <cstdint>
#include <experimental/filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <functional>
#include <utility>
#include <omp.h>

#include "taco/util/strings.h"
#include "taco/tensor.h"
#include <cstdlib>
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/transformations.h"
// #include "taco/codegen/codegen.h"
#include "taco/lower/lower.h"
#include "hypermapper_taco_client.h"

#define RANDOM 0
#define MTX 1

const taco::IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
// // const auto taco::ModeFormat::Dense  = taco::Modetaco::Format::taco::ModeFormat::Dense;
// const auto taco::ModeFormat::Sparse = taco::Modetaco::Format::taco::ModeFormat::Sparse;
// using namespace taco;
using namespace std;
namespace fs = std::experimental::filesystem;

double med(vector<double> vec) {
    typedef vector<int>::size_type vec_sz;

    vec_sz size = vec.size();
    if (size == 0)
        throw domain_error("median of an empty vector");

    sort(vec.begin(), vec.end());

    vec_sz mid = size/2;

    return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
}

typedef struct SuiteSparseTensors {
    SuiteSparseTensors() {
        //    auto ssTensorPath = getTacoTensorPath();
        // auto ssTensorPath = "";
        // ssTensorPath += "suitesparse/";
        auto ssTensorPath = std::getenv("SUITESPARSE_PATH");
        if(ssTensorPath == nullptr) {
            std::cout << "Set SUITESPARSE_PATH" << std::endl;
            exit(1);
        }
        if (fs::exists(ssTensorPath)) {
            for (auto& entry : fs::directory_iterator(ssTensorPath)) {
            std::string f(entry.path());
            // Check that the filename ends with .mtx.
                if (f.compare(f.size() - 4, 4, ".mtx") == 0) {
                    this->tensors.push_back(entry.path());
                }
            }
        }
    }

    std::vector<std::string> tensors;
} ssTensors;


struct UfuncInputCache {
  taco::Tensor<double> get_tensor(int num_k = 1000, float SPARSITY = 0.3) {
    taco::Tensor<double> tensor("tensor", {this->num_j, num_k}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense});
    srand(120);
    for (int j = 0; j < this->num_j; j++) {
      for (int k = 0; k < num_k; k++) {
        float rand_float = (float)rand() / (float)(RAND_MAX);
        tensor.insert({j, k}, (double)((int)(rand_float * 3 / SPARSITY)));
      }
    }
    tensor.pack();
    return tensor;
  }
  template<typename U>
  taco::Tensor<double> getMat(std::string path, U format, bool countNNZ = false, bool includeThird = false) {
    // See if the paths match.
    if (this->lastPath == path) {
      // TODO (rohany): Not worrying about whether the format was the same as what was asked for.
      return this->inputTensor;
    }

    // Otherwise, we missed the cache. Load in the target tensor and process it.
    // std::cout << path << std::endl;
    this->lastLoaded = taco::read(path, format);
    // We assign lastPath after lastLoaded so that if taco::read throws an exception
    // then lastPath isn't updated to the new path.
    this->lastPath = path;
    this->inputTensor = lastLoaded;

    this->num_i = this->inputTensor.getDimensions()[0];
    this->num_j = this->inputTensor.getDimensions()[1];

    // this->otherTensor = get_tensor(num_k);
    if (countNNZ) {
      this->nnz = 0;
#ifdef TACO_DEFAULT_INTEGER_TYPE
        for (auto& it : taco::iterate<double>(this->inputTensor)) {
#else
        for (auto val = this->inputTensor.template beginTyped<int64_t>(); val != this->inputTensor.template endTyped<int64_t>(); ++val) {
#endif
        this->nnz++;
      }
    }
    // if (includeThird) {
    //   this->thirdTensor = shiftLastMode<int64_t, int64_t>("C", this->otherTensor);
    // }
    return this->inputTensor;
  }


  template<typename U>
  std::pair<taco::Tensor<double>, taco::Tensor<double>> getUfuncInput(std::string path, U format, bool countNNZ = false, float sparsity=0.3, int num_k = 1000, bool includeThird = false) {
    // See if the paths match.
    if (this->lastPath == path) {
      // TODO (rohany): Not worrying about whether the format was the same as what was asked for.
      return std::make_pair(this->inputTensor, this->otherTensor);
    }

    // Otherwise, we missed the cache. Load in the target tensor and process it.
    std::cout << "Path:" << path << std::endl;
    this->lastLoaded = taco::read(path, format);
    this->lastLoaded.setName("loaded");
    // We assign lastPath after lastLoaded so that if taco::read throws an exception
    // then lastPath isn't updated to the new path.
    this->lastPath = path;
    this->inputTensor = lastLoaded;
    this->inputTensor.setName("loaded");

    this->num_i = this->inputTensor.getDimensions()[0];
    this->num_j = this->inputTensor.getDimensions()[1];

    this->otherTensor = get_tensor(num_k);
    if (countNNZ) {
      this->nnz = 0;
#ifdef TACO_DEFAULT_INTEGER_TYPE
        for (auto& it : taco::iterate<double>(this->inputTensor)) {
#else
        for (auto val = this->inputTensor.template beginTyped<int64_t>(); val != this->inputTensor.template endTyped<int64_t>(); ++val) {
#endif
        this->nnz++;
      }
    }
    // if (includeThird) {
    //   this->thirdTensor = shiftLastMode<int64_t, int64_t>("C", this->otherTensor);
    // }
    return std::make_pair(this->inputTensor, this->otherTensor);
  }

  template<typename U>
  taco::Tensor<double> getTensor(std::string path, U format, bool shift_dim=false, bool countNNZ = false, float sparsity=0.3, int num_k = 1000, bool includeThird = false) {
    // See if the paths match.
    if (this->lastPath == path) {
      // TODO (rohany): Not worrying about whether the format was the same as what was asked for.
      return this->inputTensor;
    }

    // Otherwise, we missed the cache. Load in the target tensor and process it.
    std::cout << "Path:" << path << std::endl;
    this->lastLoaded = taco::read(path, format);
    // std::cout << this->lastLoaded.getDimensions() << std::endl;
    // We assign lastPath after lastLoaded so that if taco::read throws an exception
    // then lastPath isn't updated to the new path.
    this->lastPath = path;
    this->inputTensor = lastLoaded;
    this->inputTensor.setName("loaded");

    this->num_i = this->inputTensor.getDimensions()[0];
    this->num_j = this->inputTensor.getDimensions()[1];
    this->num_k = this->inputTensor.getDimensions()[2];

    int last_dim = 0;
    if (shift_dim) {
        last_dim = this->inputTensor.getDimensions()[3];
    }

    taco::Tensor<double> copy("test", {this->num_i, this->num_k, last_dim}, taco::Sparse);

    // for (auto component : this->inputTensor) {

    // }

    if (countNNZ) {
      this->nnz = 0;
#ifdef TACO_DEFAULT_INTEGER_TYPE
        for (auto& it : taco::iterate<double>(this->inputTensor)) {
#else
        for (auto val = this->inputTensor.template beginTyped<int64_t>(); val != this->inputTensor.template endTyped<int64_t>(); ++val) {
#endif
        this->nnz++;
      }
    }
    // if (includeThird) {
    //   this->thirdTensor = shiftLastMode<int64_t, int64_t>("C", this->otherTensor);
    // }
    return this->inputTensor;
  }

  float get_sparsity() { return 1.0f - ((float)nnz) / ((float)num_i * num_j); }

  taco::Tensor<double> lastLoaded;
  std::string lastPath;

  taco::Tensor<double> inputTensor;
  taco::Tensor<double> otherTensor;
//   taco::Tensor<int64_t> thirdTensor;
  int num_i;
  int num_j;
  int num_k;
  int num_l;
  int num_m;
  int nnz;
};
UfuncInputCache inputCache;

class tacoOp {
public:
    double compute_time;
    double default_compute_time;
    std::string matrix_name;
    LoopReordering<taco::IndexVar>* reorderings;
    tacoOp() : compute_time{0.0}, default_compute_time{0.0} {}
    ~tacoOp() { delete reorderings; }
    virtual void initialize_data(int mode=RANDOM) = 0;
    virtual void compute(bool default_config=false) = 0;
    double get_compute_time() { return compute_time; }
    double get_default_compute_time() { return default_compute_time; }
    void compute_reordering(std::vector<taco::IndexVar>& ordering) {
        // TODO: Figure out how to reuse the same ordering addresses
        // if(reorder_initialized)
        //     return;
        // TODO: Remove
        // return;
        reorderings = new LoopReordering<taco::IndexVar>(ordering);
        reorderings->compute_permutations();
    }
    std::pair<taco::Tensor<double>, taco::Tensor<double>> load_tensor(std::string tensorPath, int num_k = 1000, float sparsity=0.3) {
        auto suitesparsePath = std::getenv("SUITESPARSE_PATH");
        if(suitesparsePath == nullptr) {
            std::cout << "Set SUITESPARSE_PATH" << std::endl;
            exit(1);
        }
        // std::string fullpath = std::string(suitesparsePath) + "/" + tensorPath;
        return inputCache.getUfuncInput(tensorPath, taco::CSR, true, sparsity, num_k);
    }
    std::vector<taco::IndexVar> get_reordering(int index) {
        std::vector<taco::IndexVar> temp(reorderings->get_reordering(index));
        // delete reorderings;
        return temp;
    }
    std::vector<taco::IndexVar> get_reordering(std::vector<int> reordering) {
        // std::vector<taco::IndexVar> temp(reorderings->get_reordering(reordering));
    }
    void get_reordering(std::vector<taco::IndexVar>& reordering_out, std::vector<int> reordering) {
        (reorderings->get_reordering(reordering_out, reordering));
    }
};

class SpMV : public tacoOp {
public:
    int NUM_I;
    int NUM_J;
    int NUM_K;
    float SPARSITY;
    bool initialized;
    bool reorder_initialized;
    bool cold_run;
    taco::Tensor<double> B;
    taco::Tensor<double> c;
    taco::Tensor<double> a;
    taco::util::Timer timer;
    taco::IndexStmt stmt;
    taco::IndexVar i0, i1, i10, i11, kpos, kpos0, kpos1, j0, j1;
    int run_mode;
    SpMV(int mode, int NUM_I = 10000, int NUM_J = 10000, float SPARSITY = .3) : NUM_I{NUM_I},
                                                                      NUM_J{NUM_J},
                                                                      SPARSITY{SPARSITY},
                                                                      initialized{false},
                                                                      reorder_initialized{false},
                                                                      cold_run{true},
                                                                      B("B", {NUM_I, NUM_J}, taco::CSR),
                                                                      c("c", {NUM_J}, taco::Format{taco::ModeFormat::Dense}),
                                                                      a("a", {NUM_I}, taco::Format{taco::ModeFormat::Dense}),
                                                                      i0("i0"), i1("i1"), i10("i10"), i11("i11"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1") {}
    SpMV() : run_mode(1), initialized{false},
             reorder_initialized{false},
             cold_run{true},
             i0("i0"), i1("i1"), i10("i10"), i11("i11"), j0("j0"), j1("j1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1") {}
    void initialize_data(int mode = RANDOM) override
    {
        using namespace taco;
        //TODO: Implement read from matrix market mode
        if (initialized)
            return;

        if(matrix_name != "random") {
            ssTensors mtxTensors;
            int NUM_K = 2;
            if (matrix_name == "auto") {
                std::tie(B, c) = load_tensor(mtxTensors.tensors[0], NUM_K);
            } else {
                auto ssPath = std::getenv("SUITESPARSE_PATH");
                string ssPathStr = std::string(ssPath);
                char sep = '/';
                std::string matrix_path;
                if (ssPathStr[ssPathStr.length()] == sep) {
                    matrix_path = ssPathStr + matrix_name;
                } else {
                    matrix_path = ssPathStr + "/" + matrix_name;
                }
                std::tie(B, c) = load_tensor(matrix_path, NUM_K);
            }
            NUM_I = B.getDimensions()[0];
            NUM_J = B.getDimensions()[1];
        }
        else {
            srand(120);
            for (int i = 0; i < NUM_I; i++)
            {
                for (int j = 0; j < NUM_J; j++)
                {
                    float rand_float = (float)rand() / (float)(RAND_MAX);
                    if (rand_float < SPARSITY)
                    {
                        B.insert({i, j}, (double)((int)(rand_float * 3 / SPARSITY)));
                    }
                }
            }

            for (int j = 0; j < NUM_J; j++)
            {
                float rand_float = (float)rand() / (float)(RAND_MAX);
                c.insert({j}, (double)((int)(rand_float * 3 / SPARSITY)));
            }

            c.pack();
            B.pack();
        }

        taco::Tensor<double> c_temp({NUM_J}, dense);
        for (int j = 0; j < NUM_J; j++)
        {
            float rand_float = (float)rand() / (float)(RAND_MAX);
            c_temp.insert({j}, (double)((int)(rand_float * 3 / SPARSITY)));
        }

        c = c_temp;

        c.pack();
        B.pack();

        // taco::util::Timer timer;
        // timer.start();
        // timer.stop();
        // std::cout << "Time: " << timer.getResult().mean << std::endl;

        // std::vector<taco::IndexVar> reorder_{i0, i1, j};
        std::vector<taco::IndexVar> reorder_{i0, i10, i11, j1, j0};
        compute_reordering(reorder_);
        // Avoid duplicate reinitialize
        initialized = true;
    }

    float get_sparsity() { return (run_mode == 0) ? SPARSITY : inputCache.get_sparsity(); }
    int get_num_i() { return NUM_I; }
    int get_num_j() { return NUM_J; }

    taco::IndexStmt schedule(taco::IndexStmt &sched, std::vector<int> order, int CHUNK_SIZE=16, int CHUNK_SIZE2=8, int CHUNK_SIZE3=8, int omp_scheduling_type=0, int omp_chunk_size=1, int num_threads=32) {
        using namespace taco;
        std::vector<taco::IndexVar> reorder; //= get_reordering(order);
        reorder.reserve(order.size());
        get_reordering(reorder, order);
        taco::taco_set_num_threads(num_threads);
        if(omp_scheduling_type == 0) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Static, omp_chunk_size);
        }
        else if(omp_scheduling_type == 1) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Dynamic, omp_chunk_size);
        }
        return sched.split(i, i0, i1, CHUNK_SIZE)
            .split(i1, i10, i11, CHUNK_SIZE2)
            .split(j, j0, j1, CHUNK_SIZE3)
            .reorder(reorder)
            .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
            .parallelize(j1, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces);
    }

    taco::IndexStmt schedule_(std::vector<int> order, int CHUNK_SIZE=16, int SPLIT=0, int CHUNK_SIZE2=8) {
        using namespace taco;
        if(SPLIT) {
            std::vector<taco::IndexVar> reorder = get_reordering(order);
            return stmt.split(i, i0, i1, CHUNK_SIZE)
            .split(i1, i10, i11, CHUNK_SIZE2)
            .reorder(reorder)
            .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
        }
        // std::vector<taco::IndexVar> reorder = get_reordering(order);
        std::vector<taco::IndexVar> reorder;
        reorder.reserve(order.size());
        // auto reorder = get_reordering(order);
        get_reordering(reorder, order);
        // std::cout << reorder << std::endl;
        // exit(0);
        return stmt.split(i, i0, i1, CHUNK_SIZE)
                .reorder(reorder)
                .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
    }
    taco::IndexStmt schedule2(int CHUNK_SIZE=16, int SPLIT=0, int CHUNK_SIZE2=8, int order=0) {
        using namespace taco;
        if(SPLIT) {
            std::vector<taco::IndexVar> reorder = get_reordering(order);
            return stmt.split(i, i0, i1, CHUNK_SIZE)
            .split(i1, i10, i11, CHUNK_SIZE2)
            .reorder(reorder)
            .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
        }
        std::vector<taco::IndexVar> reorder = get_reordering(order);
        return stmt.split(i, i0, i1, CHUNK_SIZE)
                .reorder(reorder)
                .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
    }

    taco::IndexStmt schedule(int CHUNK_SIZE=16, int SPLIT=0, int CHUNK_SIZE2=8, int order=0) {
        using namespace taco;
        if(SPLIT) {
            std::vector<taco::IndexVar> reorder = get_reordering(order);
            return stmt.split(i, i0, i1, CHUNK_SIZE)
            .split(i1, i10, i11, CHUNK_SIZE2)
            .reorder(reorder)
            .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
        }
        std::vector<taco::IndexVar> reorder = get_reordering(order);
        return stmt.split(i, i0, i1, CHUNK_SIZE)
                .reorder(reorder)
                .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
    }

    void generate_schedule(int chunk_size, int split, int chunk_size2, std::vector<int> order) {
        a(i) = B(i,j) * c(j);

        stmt = a.getAssignment().concretize();
        // stmt = schedule(order, chunk_size, split, chunk_size2);
    }
    void generate_schedule(int chunk_size, int split, int chunk_size2, int order) {
        a(i) = B(i,j) * c(j);

        stmt = a.getAssignment().concretize();
        stmt = schedule(chunk_size, split, chunk_size2, order);
    }

    void compute_cold_run(taco::Tensor<double> &result, taco::IndexStmt & sched) {
        result.setAssembleWhileCompute(true);
        result.compile(sched);
        // result.assemble();
        result.compute();
    }

    void compute(bool default_config = false) override {
        if(cold_run) {
            // compute_cold_run();
            timer.clear_cache();
            cold_run = false;
        }

        a(i) = B(i,j) * c(j);
        a.compile(stmt);
        a.assemble();
        timer.start();
        a.compute();
        timer.stop();

        compute_time = timer.getResult().mean;
        if(default_config) {
            default_compute_time = timer.getResult().mean;
        }
        // cold_run = true;
    }

    double compute_unscheduled() {
        // return 0.0f;
        taco::Tensor<double> result({NUM_I}, taco::dense);
        result(i) = B(i, j) * c(j);
        taco::util::Timer timer;
        result.compile();
        result.assemble();
        timer.start();
        result.compute();
        timer.stop();
        return timer.getResult().mean;
    }

    void set_cold_run() { cold_run = true; }

    void schedule_and_compute(taco::Tensor<double> &result, int chunk_size, int chunk_size2, int chunk_size3, std::vector<int> order, int omp_scheduling_type=0, int omp_chunk_size=0, int num_threads=32, bool default_config=false, int num_reps=20) {
        result(i) = B(i,j) * c(j);

        // taco::Tensor<double> temp_result()

        // std::cout << "Inside schedule and compute" << std::endl;
        // for(int l : order) {
        //     std::cout << l << " ";
        // }
        std::cout << "computing\n";
        taco::IndexStmt sched = result.getAssignment().concretize();

        // sched = schedule(sched, order, chunk_size, unroll_factor, omp_scheduling_type, omp_chunk_size, num_threads);
        sched = schedule(sched, order, chunk_size, chunk_size2, chunk_size3, omp_scheduling_type, omp_chunk_size, num_threads);

        if(cold_run) {
            taco::Tensor<double> temp_result({NUM_I}, taco::dense);
            for(int i = 0; i < 2; i++) {
                compute_cold_run(temp_result, sched);
            }
            cold_run = false;
        }


        taco::util::Timer timer;

        std::vector<double> compute_times;

        // result.setAssembleWhileCompute(true);
        timer.clear_cache();
        result.compile(sched);
        result.setNeedsAssemble(true);
        // result.assemble();
        for(int i = 0; i < num_reps; i++) {
            result.setNeedsCompute(true);
            // result.setNeedsAssemble(true);
            timer.start();
            result.assemble();
            result.compute();
            timer.stop();

            double temp_compute_time = timer.getResult().mean;

            compute_times.push_back(temp_compute_time);
            if(temp_compute_time > 1000) {
                break;
            }
        }
        compute_time = med(compute_times);

        // timer.stop();
        // compute_time = timer.getResult().mean;
        if(default_config) {
            default_compute_time = timer.getResult().mean;
        }
        timer.clear_cache();
    }

};

void printToCout(taco::IndexStmt stmt) {
//   using namespace taco;
//   std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
//   ir::Stmt compute = lower(stmt, "compute", false, true);
//   codegen->compile(compute, true);
}

class SpMM : public tacoOp {
public:
    int run_mode;
    int NUM_I;
    int NUM_J;
    int NUM_K;
    float SPARSITY;
    bool initialized;
    bool cold_run;
    taco::Tensor<double> A;
    taco::Tensor<double> B;
    taco::Tensor<double> C;
    taco::Tensor<double> expected;
    taco::IndexStmt stmt;
    taco::IndexVar i0, i1, kbounded, k0, k1, jpos, jpos0, jpos1, j0, j1;
    SpMM(int mode, int NUM_I = 10000, int NUM_J = 10000, int NUM_K = 1000, float SPARSITY = .3) : run_mode(0),
                                                                                        NUM_I{NUM_I},
                                                                                        NUM_J{NUM_J},
                                                                                        NUM_K{NUM_K},
                                                                                        SPARSITY{SPARSITY},
                                                                                        initialized{false},
                                                                                        cold_run{true},
                                                                                        A("A", {NUM_I, NUM_K}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense}),
                                                                                        B("B", {NUM_I, NUM_J}, taco::CSR),
                                                                                        C("C", {NUM_J, NUM_K}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense}),
                                                                                        expected("expected", {NUM_I, NUM_K}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense}),
                                                                                        i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1"), j0("j0"), j1("j1")
    {
    }
    SpMM()
        : run_mode(1), initialized{false}, cold_run{true},
          i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"),
          jpos("jpos"), jpos0("jpos0"), jpos1("jpos1"), j0("j0"), j1("j1") {}

    float get_sparsity() { return (run_mode == 0) ? SPARSITY : inputCache.get_sparsity(); }
    int get_num_i() { return NUM_I; }
    int get_num_j() { return NUM_J; }
    void initialize_data(int mode = RANDOM) override
    {
        using namespace taco;
        if (initialized)
            return;

        if (NUM_K == 0) {
          cout << "manually set spmm_handler->NUM_K" << endl;
          exit(1);
        }

        if (matrix_name != "random") {
            ssTensors mtxTensors;
            if (matrix_name == "auto") {
                std::tie(B, C) = load_tensor(mtxTensors.tensors[0], NUM_K);
            } else {
                auto ssPath = std::getenv("SUITESPARSE_PATH");
                string ssPathStr = std::string(ssPath);
                char sep = '/';
                std::string matrix_path;
                if (ssPathStr[ssPathStr.length()] == sep) {
                    matrix_path = ssPathStr + matrix_name;
                } else {
                    matrix_path = ssPathStr + "/" + matrix_name;
                }
                std::tie(B, C) = load_tensor(matrix_path, NUM_K);
            }
            NUM_I = B.getDimensions()[0];
            NUM_J = B.getDimensions()[1];


            // A.pack();
            // std::cout << A << std::endl;
        }
        else {
            srand(120);
            for (int i = 0; i < NUM_I; i++)
            {
                for (int j = 0; j < NUM_J; j++)
                {
                    float rand_float = (float)rand() / (float)(RAND_MAX);
                    if (rand_float < SPARSITY)
                    {
                        B.insert({i, j}, (double)((int)(rand_float * 3 / SPARSITY)));
                    }
                }
            }

            for (int j = 0; j < NUM_J; j++)
            {
                for (int k = 0; k < NUM_K; k++)
                {
                    float rand_float = (float)rand() / (float)(RAND_MAX);
                    C.insert({j, k}, (double)((int)(rand_float * 3 / SPARSITY)));
                }
            }
        }
        // taco::Tensor<double> result("A", {NUM_I, NUM_K}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense});
        // A = result;

        B.pack();
        C.pack();
        // A(i, k) = B(i, j) * C(j, k);
        std::vector<taco::IndexVar> reorder_{i0, i1, jpos0, k, jpos1};
        compute_reordering(reorder_);
        // Avoid duplicate reinitialize
        initialized = true;
    }

    bool check_correctness(taco::Tensor<double> &actual) {
        return equals(expected, actual);
    }

    taco::IndexStmt schedule(std::vector<int> order, int chunk_size=16, int unroll_factor=8, int omp_scheduling_type=0, int omp_chunk_size=1, int num_threads=32) {
        using namespace taco;
        std::vector<taco::IndexVar> reorder; //= get_reordering(order);
        reorder.reserve(order.size());
        get_reordering(reorder, order);
        taco::taco_set_num_threads(num_threads);
        if(omp_scheduling_type == 0) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Static, omp_chunk_size);
        }
        else if(omp_scheduling_type == 1) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Dynamic, omp_chunk_size);
        }
        return stmt.split(i, i0, i1, chunk_size)
                .pos(j, jpos, B(i,j))
                .split(jpos, jpos0, jpos1, unroll_factor)
                .reorder(reorder)
                .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
                .parallelize(k, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces);
    }

    taco::IndexStmt schedule(taco::IndexStmt &sched, std::vector<int> order, int chunk_size=16, int unroll_factor=8, int omp_scheduling_type=0, int omp_chunk_size=1, int num_threads=32) {
        using namespace taco;
        std::vector<taco::IndexVar> reorder; //= get_reordering(order);
        reorder.reserve(order.size());
        get_reordering(reorder, order);
        taco::taco_set_num_threads(num_threads);
        // omp_set_schedule(omp_sched_dynamic, 32);
        // std::cout << getenv("OMP_SCHEDULE") << std::endl;
        // std::cout << chunk_size << ", " << unroll_factor << ", " << omp_scheduling_type << ", " << omp_chunk_size << ", " << num_threads << std::endl;

        if(omp_scheduling_type == 0) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Static, omp_chunk_size);
        }
        else if(omp_scheduling_type == 1) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Dynamic, omp_chunk_size);
        }
        return sched.split(i, i0, i1, chunk_size)
                .pos(j, jpos, B(i,j))
                .split(jpos, jpos0, jpos1, unroll_factor)
                .reorder(reorder)
                .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
                .parallelize(k, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces);
    }

    taco::IndexStmt schedule2(std::vector<int> order, int chunk_size=16, int unroll_factor=8, int omp_scheduling_type=0, int omp_chunk_size=1, int num_threads=32) {
        using namespace taco;
        std::vector<taco::IndexVar> reorder; //= get_reordering(order);
        reorder.reserve(order.size());
        get_reordering(reorder, order);
        taco::taco_set_num_threads(num_threads);
        if(omp_scheduling_type == 0) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Static, omp_chunk_size);
        }
        else if(omp_scheduling_type == 1) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Dynamic, omp_chunk_size);
        }
        return stmt.split(i, i0, i1, chunk_size)
                .pos(j, jpos, B(i,j))
                .split(jpos, jpos0, jpos1, unroll_factor)
                .reorder(reorder)
                .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
                .parallelize(jpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
    }
    taco::IndexStmt schedule(int chunk_size=16, int unroll_factor=8, int order=0) {
        using namespace taco;
        std::vector<taco::IndexVar> reorder = get_reordering(order);
        return stmt.split(i, i0, i1, chunk_size)
                .pos(j, jpos, B(i,j))
                .split(jpos, jpos0, jpos1, unroll_factor)
                .reorder(reorder)
                .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
                .parallelize(k, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces);
    }

    // THIS IS THE RELEVANT ONE
    void generate_schedule(taco::Tensor<double> &result, int chunk_size, int unroll_factor, std::vector<int> order, int omp_scheduling_type, int omp_chunk_size, int num_threads) {
        result(i, k) = B(i, j) * C(j, k);

        // taco::taco_set_num_threads(num_threads);
        stmt = result.getAssignment().concretize();
        // std::vector<int> order_{0,1,2,3,4};
        // stmt = schedule(order_, chunk_size, unroll_factor, omp_scheduling_type, omp_chunk_size);
        stmt = schedule(order, chunk_size, unroll_factor, omp_scheduling_type, omp_chunk_size, num_threads);
        // std::cout << stmt << std::endl;
    }

    void compute_cold_run() {
        A.compile(stmt);
        A.assemble();
        A.compute();
    }

    void compute_cold_run(taco::Tensor<double> &result) {
        result.compile(stmt);
        result.assemble();
        result.compute();
    }

    void compute_cold_run(taco::Tensor<double> &result, taco::IndexStmt & sched) {
        result.setAssembleWhileCompute(true);
        result.compile(sched);
        // result.assemble();
        result.compute();
    }

    void set_cold_run() { cold_run = true; }

    taco::IndexStmt schedule3(taco::IndexStmt &sched, std::vector<int> order, int chunk_size=16, int unroll_factor=8, int omp_scheduling_type=0, int omp_chunk_size=1, int num_threads=32) {
        using namespace taco;
        std::vector<taco::IndexVar> reorder; //= get_reordering(order);
        reorder.reserve(order.size());
        get_reordering(reorder, order);
        taco::taco_set_num_threads(num_threads);
        // omp_set_schedule(omp_sched_dynamic, 32);
        // std::cout << getenv("OMP_SCHEDULE") << std::endl;
        // std::cout << chunk_size << ", " << unroll_factor << ", " << omp_scheduling_type << ", " << omp_chunk_size << ", " << num_threads << std::endl;

        if(omp_scheduling_type == 0) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Static, omp_chunk_size);
        }
        else if(omp_scheduling_type == 1) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Dynamic, omp_chunk_size);
        }

        int chunk_size2 = 2;
        int chunk_size3 = 32;
        std::vector<taco::IndexVar> reorder_{i1, j1, k1, i0, j0, k0};
        return sched.split(i, i1, i0, 4)
                .split(j, j1, j0, chunk_size2)
                .split(k, k1, k0, chunk_size3)
                .reorder({i1, j1, k1, i0, j0, k0})
                .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
                // .parallelize(j, ParallelUnit::CPUThread, OutputRaceStrategy::IgnoreRaces)
                // .parallelize(k, ParallelUnit::CPUThread, OutputRaceStrategy::IgnoreRaces);
    }

    void schedule_and_compute(taco::Tensor<double> &result, int chunk_size, int unroll_factor, std::vector<int> order, int omp_scheduling_type=0, int omp_chunk_size=0, int num_threads=32, bool default_config=false, int num_reps=20) {
        result(i, k) = B(i, j) * C(j, k);

        // taco::Tensor<double> temp_result()

        taco::IndexStmt sched = result.getAssignment().concretize();

        sched = schedule(sched, order, chunk_size, unroll_factor, omp_scheduling_type, omp_chunk_size, num_threads);

        if(cold_run) {
	          taco::Tensor<double> temp_result({NUM_I, NUM_K}, taco::dense);
            for(int i = 0; i < 2; i++) {
                compute_cold_run(temp_result, sched);
            }
            cold_run = false;
        }


        taco::util::Timer timer;

        std::vector<double> compute_times;

        // // result.setAssembleWhileCompute(true);

        timer.clear_cache();
        result.compile(sched);
        result.setNeedsAssemble(true);
        result.assemble();
        for(int i = 0; i < num_reps; i++) {
            timer.start();
            result.setNeedsCompute(true);
            result.compute();
            timer.stop();

            double temp_compute_time = timer.getResult().mean;

            compute_times.push_back(temp_compute_time);
            if(temp_compute_time > 10000) {
                break;
            }
        }
        compute_time = med(compute_times);

        // timer.stop();
        // compute_time = timer.getResult().mean;
        if(default_config) {
            default_compute_time = timer.getResult().mean;
        }
        timer.clear_cache();
    }

    void compute(bool default_config = false)
    {
        if(cold_run) {
            // std::cout << "Computing cold run" << std::endl;

            for(int i = 0; i < 1; i++)
                compute_cold_run();
            cold_run = false;
        }
        taco::util::Timer timer;
        timer.clear_cache();

        A(i, k) = B(i, j) * C(j, k);
        A.compile(stmt);
        printToCout(stmt);
        A.assemble();
        timer.start();
        A.compute();
        timer.stop();

        compute_time = timer.getResult().mean;
        if (default_config)
        {
            default_compute_time = timer.getResult().mean;
        }
        timer.clear_cache();
    }

    double compute_unscheduled() {
        taco::Tensor<double> result({NUM_I, NUM_K}, taco::dense);
        result(i, k) = B(i, j) * C(j, k);
        taco::util::Timer timer;
        result.compile();
        result.assemble();
        timer.start();
        result.compute();
        timer.stop();
        return timer.getResult().mean;
    }

    void compute(taco::Tensor<double> &result, bool default_config = false)
    {
        if(cold_run) {
            // std::cout << "Computing cold run" << std::endl;

            for(int i = 0; i < 5; i++)
                compute_cold_run(result);
            cold_run = false;
        }
        taco::util::Timer timer;
        result(i, k) = B(i, j) * C(j, k);
        result.compile(stmt);
        result.assemble();
        timer.start();
        result.compute();
        timer.stop();
        compute_time = timer.getResult().mean;
        if (default_config)
        {
            default_compute_time = timer.getResult().mean;
        }
        timer.clear_cache();
    }

    taco::Tensor<double> get_A() {
        taco::Tensor<double> result({NUM_I, NUM_K}, taco::dense);
        return result;
    }

    taco::Tensor<double> get_B() { return B; }
};

class SDDMM : public tacoOp {
public:
    int NUM_I;
    int NUM_J;
    int NUM_K;
    float SPARSITY;
    bool initialized;
    bool cold_run;
    taco::Tensor<double> A;
    taco::Tensor<double> B;
    taco::Tensor<double> C;
    taco::Tensor<double> D;
    taco::IndexStmt stmt;
    // taco::IndexVar i0, i1, kpos, kpos0, kpos1;
    taco::IndexVar i0, i1, jpos, jpos0, jpos1;
    int run_mode;
    float get_sparsity() { return (run_mode == 0) ? SPARSITY : inputCache.get_sparsity(); }
    int get_num_i() { return NUM_I; }
    int get_num_j() { return NUM_J; }
    SDDMM(int mode, int NUM_I = 10000, int NUM_J = 10000, int NUM_K = 1000, float SPARSITY = .3) : NUM_I{NUM_I},
                                                                                         NUM_J{NUM_J},
                                                                                         NUM_K{NUM_K},
                                                                                         SPARSITY{SPARSITY},
                                                                                         initialized{false},
                                                                                         cold_run{true},
                                                                                         A("A", {NUM_I, NUM_J}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense}),
                                                                                         B("B", {NUM_I, NUM_J}, taco::CSR),
                                                                                         C("C", {NUM_I, NUM_K}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense}),
                                                                                         D("D", {NUM_K, NUM_J}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense}),
                                                                                        //  i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1"), run_mode(0)
                                                                                         i0("i0"), i1("i1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1"), run_mode(0)
    {
    }
    SDDMM() : run_mode(1), initialized{false}, cold_run{true},
            //   i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1") {}
              i0("i0"), i1("i1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1") {}
    void initialize_data(int mode = RANDOM) override
    {
        //TODO: Implement read from matrix market mode
        using namespace taco;
        // Sanity check
        if (initialized)
            return;

        srand(268238);
        if(matrix_name != "random") {
            ssTensors mtxTensors;
            if (matrix_name == "auto") {
                B = inputCache.getMat(mtxTensors.tensors[0], taco::CSR, true);
                std::cout << mtxTensors.tensors[0] << endl;
                B.pack();
            } else {
                // get the suitesparse path and merge it with the matrx name.
                // Add "/" in between if not included in suitesparse path.
                auto ssPath = std::getenv("SUITESPARSE_PATH");
                string ssPathStr = std::string(ssPath);
                char sep = '/';
                std::string matrix_path;
                if (ssPathStr[ssPathStr.length() - 1] == sep) {
                    matrix_path = ssPathStr + matrix_name;
                } else {
                    matrix_path = ssPathStr + "/" + matrix_name;
                }
                B = inputCache.getMat(matrix_path, taco::CSR, true, NUM_K);
                std::cout << matrix_path << endl;
                B.pack();
                // std::cout << B << std::endl;
                // exit(1);
            }
            NUM_I = B.getDimension(0);
            NUM_J = B.getDimension(1);
            NUM_K = 256;
        }
        else {
            for (int i = 0; i < NUM_I; i++)
            {
                for (int j = 0; j < NUM_J; j++)
                {
                    float rand_float = (float)rand() / (float)(RAND_MAX);
                    if (rand_float < SPARSITY)
                    {
                        B.insert({i, j}, (double)((int)(rand_float * 3 / SPARSITY)));
                    }
                }
            }
        }

        taco::Tensor<double> C_tmp({NUM_I, NUM_K}, dense);
        for (int i = 0; i < NUM_I; i++)
        {
            for (int k = 0; k < NUM_K; k++)
            {
                float rand_float = (float)rand() / (float)(RAND_MAX);
                C_tmp.insert({i, k}, (double)((int)(rand_float * 3 / SPARSITY)));
            }
        }
        C = C_tmp;

        taco::Tensor<double> D_tmp({NUM_K, NUM_J}, dense);
        for (int k = 0; k < NUM_K; k++)
        {
            for (int j = 0; j < NUM_J; j++)
            {
                float rand_float = (float)rand() / (float)(RAND_MAX);
                D_tmp.insert({k, j}, (double)((int)(rand_float * 3 / SPARSITY)));
            }
        }
        D = D_tmp;
        taco::Tensor<double> result("A", {NUM_I, NUM_J}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense});
        A = result;

        B.pack();
        C.pack();
        D.pack();
        B.setName("loadedMat");

        // A(i,j) = B(i,j) * C(i,k) * D(k,j);

        cout << "Matrix dimensions" << endl;
        cout << "A: [" << A.getDimensions()[0] << "," << A.getDimensions()[1] << "]" << endl;
        cout << "B: [" << B.getDimensions()[0] << "," << B.getDimensions()[1] << "]" << endl;
        cout << "C: [" << C.getDimensions()[0] << "," << C.getDimensions()[1] << "]" << endl;
        cout << "D: [" << D.getDimensions()[0] << "," << D.getDimensions()[1] << "]" << endl;

        // Avoid duplicate reinitialize
        initialized = true;
        std::vector<taco::IndexVar> reorder_{i0, i1, jpos0, k, jpos1};
        // std::vector<taco::IndexVar> reorder_{i0, i1, kpos0, j, kpos1};
        compute_reordering(reorder_);
    }
    // float get_sparsity() { return (matrix_name == "random") ? SPARSITY : inputCache.get_sparsity(); }
    // int get_num_j() { return NUM_J; }

    void compute_cold_run() {
        A.compile(stmt);
        A.assemble();
        A.compute();
    }

    void compute_cold_run(taco::Tensor<double> &result, taco::IndexStmt &sched) {
        // A(i,j) = B(i,j) * C(i,k) * D(k,j);
        result.compile(sched);
        result.assemble();
        result.compute();
    }

    taco::IndexStmt schedule(int chunk_size=16, int unroll_factor=8, int order=0) {
        //TODO: Unroll factor needs to be less than the chunk size
        using namespace taco;
        std::vector<taco::IndexVar> reorder = get_reordering(order);
        return stmt.split(i, i0, i1, chunk_size)
                .pos(j, jpos, B(i,j))
                .split(jpos, jpos0, jpos1, unroll_factor)
                .reorder(reorder)
                .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
                .parallelize(jpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
    }

    taco::IndexStmt schedule(std::vector<int> order, int chunk_size=16, int unroll_factor=8, int omp_scheduling_type=0, int omp_chunk_size=0) {
        using namespace taco;
        // std::vector<taco::IndexVar> reorder = get_reordering(order);
        std::vector<taco::IndexVar> reorder; //= get_reordering(order);
        reorder.reserve(order.size());
        if(omp_scheduling_type == 0) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Static, omp_chunk_size);
        }
        else if(omp_scheduling_type == 1) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Dynamic, omp_chunk_size);
        }
        get_reordering(reorder, order);
        // taco::taco_set_parallel_schedule(taco::ParallelSchedule::Dynamic, 16);
        return stmt.split(i, i0, i1, chunk_size)
                .pos(j, jpos, B(i,j))
                .split(jpos, jpos0, jpos1, unroll_factor)
                // .pos(k, kpos, B(i,j))
                // .split(kpos, kpos0, kpos1, UNROLL_FACTOR)
                .reorder(reorder)
                .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
                .parallelize(jpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
    }

    taco::IndexStmt schedule(taco::IndexStmt &sched, std::vector<int> order, int chunk_size=16, int unroll_factor=8, int omp_scheduling_type=0, int omp_chunk_size=0, int omp_num_threads=32) {
        using namespace taco;
        // std::vector<taco::IndexVar> reorder = get_reordering(order);
        std::vector<taco::IndexVar> reorder; //= get_reordering(order);
        taco::taco_set_num_threads(omp_num_threads);
        reorder.reserve(order.size());
        if(omp_scheduling_type == 0) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Static, omp_chunk_size);
        }
        else if(omp_scheduling_type == 1) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Dynamic, omp_chunk_size);
        }
        get_reordering(reorder, order);
        return sched.split(i, i0, i1, chunk_size)
                .pos(j, jpos, B(i,j))
                .split(jpos, jpos0, jpos1, unroll_factor)
                // .pos(k, kpos, B(i,j))
                // .split(kpos, kpos0, kpos1, UNROLL_FACTOR)
                .reorder(reorder)
                .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
                .parallelize(jpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
    }

    void generate_schedule(int chunk_size, int unroll_factor, int order) {
        // A(i,k) = B(i,k) * C(i,j) * D(j,k);
        A(i,j) = B(i,j) * C(i,k) * D(k,j);

        stmt = A.getAssignment().concretize();
        stmt = schedule(chunk_size, unroll_factor, order);
    }

    void generate_schedule(std::vector<int> order, int chunk_size, int unroll_factor) {
        // A(i,k) = B(i,k) * C(i,j) * D(j,k);
        A(i,j) = B(i,j) * C(i,k) * D(k,j);

        stmt = A.getAssignment().concretize();
        stmt = schedule(order, chunk_size, unroll_factor);
    }

    void generate_schedule(int chunk_size, int unroll_factor, std::vector<int> order, int omp_scheduling_type=0, int omp_chunk_size=0, int num_threads=32) {
        // A(i,k) = B(i,k) * C(i,j) * D(j,k);
        A(i,j) = B(i,j) * C(i,k) * D(k,j);

        taco::taco_set_num_threads(num_threads);
        // if(omp_scheduling_type == 0) {
        //     taco::taco_set_parallel_schedule(taco::ParallelSchedule::Static, omp_chunk_size);
        // }
        // else if(OMP_SCHEDULING_TYPE == 1) {
        //     taco::taco_set_parallel_schedule(taco::ParallelSchedule::Dynamic, omp_chunk_size);
        // }

        stmt = A.getAssignment().concretize();
        stmt = schedule(order, chunk_size, unroll_factor, omp_scheduling_type, omp_chunk_size);
    }


    void generate_schedule(taco::Tensor<double>& result, int chunk_size, int unroll_factor, std::vector<int> order, int omp_scheduling_type=0, int omp_chunk_size=0, int num_threads=32) {
        // A(i,k) = B(i,k) * C(i,j) * D(j,k);
        result(i,j) = B(i,j) * C(i,k) * D(k,j);

        taco::taco_set_num_threads(num_threads);
        // if(omp_scheduling_type == 0) {
        //     taco::taco_set_parallel_schedule(taco::ParallelSchedule::Static, omp_chunk_size);
        // }
        // else if(OMP_SCHEDULING_TYPE == 1) {
        //     taco::taco_set_parallel_schedule(taco::ParallelSchedule::Dynamic, omp_chunk_size);
        // }

        taco::IndexStmt stmt = result.getAssignment().concretize();
        stmt = schedule(order, chunk_size, unroll_factor, omp_scheduling_type, omp_chunk_size);
    }

    void set_cold_run() { cold_run = true; }

    void schedule_and_compute(taco::Tensor<double> &result, int chunk_size, int unroll_factor, std::vector<int> order, int omp_scheduling_type=0, int omp_chunk_size=0, int num_threads=32, bool default_config=false, int num_reps=20) {
        // result(i, j) = B(i, j) * C(i, k) * D(k, j);

        // std::cout << "Elements: " << std::endl;
        // for(auto elem : order) {
        //     std::cout << elem << " ";
        // }
        // std::cout << std::endl;

        result(i, j) = B(i, j) * C(i, k) * D(k, j);
        taco::IndexStmt sched = result.getAssignment().concretize();
        sched = schedule(sched, order, chunk_size, unroll_factor, omp_scheduling_type, omp_chunk_size, num_threads);

        // if(cold_run) {
        //     for(int i = 0; i < 5; i++) {
        //         // taco::Tensor<double> temp_result({NUM_I, NUM_J}, taco::dense);
        //         compute_cold_run(result, sched);
        //     }
        //     cold_run = false;
        // }


        taco::util::Timer timer;
        std::vector<double> compute_times;


        timer.clear_cache();
        result.compile(sched);
        result.setNeedsAssemble(true);
        result.assemble();
        for(int i = 0; i < num_reps; i++) {
            timer.start();
            result.setNeedsCompute(true);
            result.compute();
            timer.stop();

            double temp_compute_time = timer.getResult().mean;

            compute_times.push_back(temp_compute_time);
            if(temp_compute_time > 10000) {
                break;
            }
        }
        compute_time = med(compute_times);

        // if(cold_run && order == std::vector<int>{0,2,3,1,4}) {
        //     std::cout << result.getSource() << std::endl;
        //     std::cout << sched << std::endl;
        //     cold_run = false;
        //     if(order == std::vector<int>{0,2,3,1,4}) {
        //         exit(0);
        //     }
        // }
        timer.stop();
        // compute_time = timer.getResult().mean;
        if(default_config) {
            default_compute_time = timer.getResult().mean;
        }
        timer.clear_cache();
    }

    double compute_unscheduled() {
        taco::Tensor<double> result({NUM_I, NUM_J}, taco::dense);
        result(i,j) = B(i,j) * C(i,k) * D(k,j);
        taco::util::Timer timer;
        result.compile();
        result.assemble();
        timer.start();
        result.compute();
        timer.stop();
        return timer.getResult().mean;
    }

    void compute(bool default_config = false)
    {
        // if(cold_run) {
        //     for(int i = 0; i < 5; i++) {
        //         compute_cold_run();
        //     }
        //     cold_run = false;
        // }

        taco::util::Timer timer;

        taco::Tensor<double> result({NUM_I, NUM_J}, taco::dense);

        // A(i,k) = B(i,k) * C(i,j) * D(j,k);
        result(i,j) = B(i,j) * C(i,k) * D(k,j);
        result.compile(stmt);
        result.assemble();
        timer.start();
        result.compute();
        timer.stop();
        std::cout << result.getSource() << std::endl;
        compute_time = timer.getResult().mean;
        if (default_config)
        {
            default_compute_time = timer.getResult().mean;
        }
        timer.clear_cache();
    }

    void compute(taco::Tensor<double>& result, bool default_config = false)
    {
        // if(cold_run) {
        //     for(int i = 0; i < 5; i++) {
        //         compute_cold_run(result);
        //     }
        //     cold_run = false;
        // }
        taco::util::Timer timer;
        // timer.clear_cache();
        result(i,j) = B(i,j) * C(i,k) * D(k,j);
        result.compile(stmt);
        std::cout << result.getSource() << std::endl;
        result.assemble();
        timer.start();
        result.compute();
        timer.stop();
        result.pack();
        compute_time = timer.getResult().mean;
        if (default_config)
        {
            default_compute_time = timer.getResult().mean;
        }
        // timer.clear_cache();
    }

};


class TTV : public tacoOp {
public:
    int NUM_I;
    int NUM_J;
    int NUM_K;
    float SPARSITY;
    bool initialized;
    bool cold_run;
    taco::Tensor<double> A;
    taco::Tensor<double> B;
    taco::Tensor<double> c;
    taco::IndexStmt stmt;
    taco::IndexVar f, fpos, chunk, fpos2, k1, k2, kpos, kpos1, kpos2, i0, i1;
    int run_mode, num_reps;
    TTV(int mode, int NUM_I = 1000, int NUM_J = 1000, int NUM_K = 1000, float SPARSITY = .3) : NUM_I{NUM_I},
                                                                                     NUM_J{NUM_J},
                                                                                     NUM_K{NUM_K},
                                                                                     SPARSITY{SPARSITY},
                                                                                     initialized{false},
                                                                                     cold_run{true},
                                                                                     A("A", {NUM_I, NUM_J}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense}),
                                                                                     B("B", {NUM_I, NUM_J, NUM_K}, taco::Format{taco::ModeFormat::Sparse, taco::ModeFormat::Sparse, taco::ModeFormat::Sparse}),
                                                                                     c("c", {NUM_K}, taco::Format{taco::ModeFormat::Dense}),
                                                                                     f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2"), k1("k1"), k2("k2"),
                                                                                     i0("i0"), i1("i1")
    {
    }
    TTV() : run_mode(1), initialized{false}, cold_run{true},
            f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2"), k1("k1"), k2("k2"), i0("i0"), i1("i1"),
            kpos("kpos"), kpos1("kpos1"), kpos2("kpos2"){}
    float get_sparsity() { return (run_mode == 0) ? SPARSITY : inputCache.get_sparsity(); }
    void set_cold_run() { cold_run = true; }
    void initialize_data(int mode = RANDOM) override
    {
        //TODO: Implement read from matrix market mode
        using namespace taco;

        // Sanity check: avoid initializing multiple times
        if (initialized)
            return;

        srand(9536);

        int nnz = 0;
        if (mode == RANDOM) {
            taco::Tensor<double> res("res", {NUM_I, NUM_J, NUM_K}, taco::Sparse);
            B = res;
            for (int i = 0; i < NUM_I; i++)
                {
                    for (int j = 0; j < NUM_J; j++)
                        {
                            for (int k = 0; k < NUM_K; k++)
                                {
                                    float rand_float = (float)rand() / (float)(RAND_MAX);
                                    if (rand_float < SPARSITY)
                                        {
                                            B.insert({i, j, k}, (double)((int)(rand_float * 3 / SPARSITY)));
                                            nnz++;
                                        }
                                }
                        }
                }
        }

        else {
            auto ssPath = std::getenv("FROST_PATH");
            if(ssPath == nullptr) {
                std::cout << "Environment variable FROST_PATH not set\n";
            }
            std::string ssPathStr = std::string(ssPath);

            char sep = '/';
            std::string matrix_path;
            if (ssPathStr[ssPathStr.length()] == sep) {
                matrix_path = ssPathStr + matrix_name;
            } else {
                matrix_path = ssPathStr + "/" + matrix_name;
            }

            B = inputCache.getTensor(matrix_path, Sparse, true);
            NUM_I = inputCache.num_i;
            NUM_J = inputCache.num_j;
            NUM_K = inputCache.num_k;
        }

        std::cout << "Dimensions: " << NUM_I << ", " << NUM_J << ", " << NUM_K << std::endl;
        std::cout << "NNZ: " << nnz << std::endl;

        taco::Tensor<double> c_("c", {NUM_K}, taco::Format{taco::ModeFormat::Dense});
        c = c_;
        for (int k = 0; k < NUM_K; k++)
        {
            float rand_float = (float)rand() / (float)(RAND_MAX);
            c.insert({k}, (double)((int)(rand_float * 3)));
        }

        B.pack();
        c.pack();

        std::vector<taco::IndexVar> reorder_{i0, chunk, fpos2, kpos1, kpos2};
        compute_reordering(reorder_);
        // Avoid duplicate reinitialize
        initialized = true;
    }
    taco::IndexStmt schedule(int CHUNK_SIZE=16, int order=0) {
        using namespace taco;
        std::vector<taco::IndexVar> reorder = get_reordering(order);
        return stmt.fuse(i, j, f)
                .pos(f, fpos, B(i,j,k))
                .split(fpos, chunk, fpos2, CHUNK_SIZE)
                .reorder(reorder)
                .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
    }


    void generate_schedule(int chunk_size, int order) {
        A(i, j) = B(i, j, k) * c(k);

        stmt = A.getAssignment().concretize();
        stmt = schedule(chunk_size, order);
    }
    void compute_cold_run(taco::Tensor<double> &result, taco::IndexStmt &sched) {
        result.compile(sched);
        result.assemble();
        result.compute();
    }
    int get_num_i() { return NUM_I; }
    int get_num_j() { return NUM_J; }

    double compute_unscheduled() {
        taco::Tensor<double> result = copyNonZeroStructure({NUM_I, NUM_J}, {taco::Sparse, taco::Sparse}, B, 2);
        result(i, j) = B(i, j, k) * c(k);
        taco::util::Timer timer;
        result.setPreserveNonZero(true);
        result.setNeedsAssemble(false);
        result.setAssembleWhileCompute(false);
        result.compile();
        // result.assemble();
        timer.start();
        result.compute();
        timer.stop();
        return timer.getResult().mean;
    }


    //taco::IndexStmt schedule(taco::IndexStmt &sched, std::vector<int> order, int chunk_size=16, int omp_scheduling_type=0, int omp_chunk_size=0, int omp_num_threads=32) {

        taco::IndexStmt schedule(taco::IndexStmt &sched, std::vector<int> order, int chunk_size_i=16, int chunk_size_fpos=16, int chunk_size_k = 16, int omp_scheduling_type=0, int omp_chunk_size=0, int omp_num_threads=32) {

        using namespace taco;
        // std::vector<taco::IndexVar> reorder = get_reordering(order);
        std::vector<taco::IndexVar> reorder; //= get_reordering(order);
        taco::taco_set_num_threads(omp_num_threads);
        reorder.reserve(order.size());
        if(omp_scheduling_type == 0) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Static, omp_chunk_size);
        }
        else if(omp_scheduling_type == 1) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Dynamic, omp_chunk_size);
        }
        get_reordering(reorder, order);
        return sched.split(i, i0, i1, chunk_size_i).fuse(i1, j, f)
            .pos(f, fpos, B(i,j,k))
            .split(fpos, chunk, fpos2, chunk_size_fpos)
            .pos(k, kpos, B(i,j,k))
            .split(kpos, kpos1, kpos2, chunk_size_k)
            .reorder(reorder)
            .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
            .parallelize(kpos2, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);


            // return stmt.fuse(i, j, f)
            //             .pos(f, fpos, B(i,j,k))
            //             .split(fpos, chunk, fpos2, chunk_size)
            //             .split(k, k1, k2, chunk_size)
            //             .reorder(reorder)
            //             .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
        // return sched.fuse(i, j, f)
        //         .pos(f, fpos, B(i,j,k))
        //         .split(fpos, chunk, fpos2, chunk_size)
        //         .reorder(reorder)
        //         .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
    }

    void schedule_and_compute(taco::Tensor<double> &result_, int chunk_size_i, int chunk_size_fpos, int chunk_size_k,
                              std::vector<int> order, int omp_scheduling_type=0, int omp_chunk_size=0, int num_threads=32, bool default_config=false,
                              int num_reps=10) {
        taco::Tensor<double> result = copyNonZeroStructure({NUM_I, NUM_J}, {taco::Sparse, taco::Sparse}, B, 2);
        result(i, j) = B(i, j, k) * c(k);

        // std::cout << "Elements: " << std::endl;
        // for(auto elem : order) {
        //     std::cout << elem << " ";
        // }
        // std::cout << std::endl;

        taco::IndexStmt sched = result.getAssignment().concretize();
        std::vector<int> order_{0,1,2};
        // sched = schedule(sched, order_, chunk_size, omp_scheduling_type, omp_chunk_size, num_threads);
        sched = schedule(sched, order, chunk_size_i, chunk_size_fpos, chunk_size_k, omp_scheduling_type, omp_chunk_size, num_threads);

        // if(cold_run) {
        //     for(int i = 0; i < 5; i++) {
        //         // taco::Tensor<double> temp_result({NUM_I, NUM_J}, taco::dense);
        //         compute_cold_run(result, sched);
        //     }
            cold_run = false;
        // }


	      taco::util::Timer timer;
        std::vector<double> compute_times;
        timer.clear_cache();
        result.setPreserveNonZero(true);
        result.setNeedsAssemble(false);
        result.compile(sched);
//        result.assemble();
        for(int i = 0; i < num_reps; i++) {
            timer.start();
            result.setNeedsCompute(true);
            result.compute();
            timer.stop();

            double temp_compute_time = timer.getResult().mean;

            compute_times.push_back(temp_compute_time);
            if(temp_compute_time > 10000) {
                break;
            }
        }
        compute_time = med(compute_times);

        // timer.stop();
        // compute_time = timer.getResult().mean;
        if(default_config) {
            default_compute_time = timer.getResult().mean;
        }
        timer.clear_cache();
    }

    void compute(bool default_config = false) override
    {
        if(cold_run) {
            // compute_cold_run(result, sched);
            cold_run = false;
        }
        taco::util::Timer timer;

        A(i, j) = B(i, j, k) * c(k);
        A.compile(stmt);
        A.assemble();
        timer.start();
        A.compute();
        timer.stop();

        compute_time = timer.getResult().mean;
        if (default_config)
        {
            default_compute_time = timer.getResult().mean;
        }
    }

};

class TTM : public tacoOp {
public:
    int NUM_I;
    int NUM_J;
    int NUM_K;
    int NUM_L;
    float SPARSITY;
    bool initialized;
    bool cold_run;
    taco::Tensor<double> A;
    taco::Tensor<double> B;
    taco::Tensor<double> C;
    taco::IndexStmt stmt;
    taco::IndexVar f, fpos, chunk, fpos2, kpos, kpos1, kpos2;
    int run_mode;
    TTM() : run_mode(1), initialized{false}, cold_run{true},
            f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2"), kpos("kpos"), kpos1("kpos1"), kpos2("kpos2"){}
    TTM(int mode, int NUM_I = 1000, int NUM_J = 1000, int NUM_K = 1000, int NUM_L = 1000, float SPARSITY = .1) : NUM_I{NUM_I},
                                                                                     NUM_J{NUM_J},
                                                                                     NUM_K{NUM_K},
                                                                                     NUM_L{NUM_L},
                                                                                     SPARSITY{SPARSITY},
                                                                                     initialized{false},
                                                                                     cold_run{true},
                                                                                     A("A", {NUM_I, NUM_J}, taco::CSR),
                                                                                     B("B", {NUM_J, NUM_K}, {taco::ModeFormat::Dense, taco::ModeFormat::Dense}),
                                                                                     C("C", {NUM_I, NUM_K}, {taco::ModeFormat::Dense, taco::ModeFormat::Dense}),
                                                                                     f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2"), kpos("kpos"), kpos1("kpos1"), kpos2("kpos2")
    {
    }
    void set_cold_run() { cold_run = true; }
    void initialize_data(int mode = RANDOM) override
    {
        //TODO: Implement read from matrix market mode
        using namespace taco;

        // Sanity check: avoid initializing multiple times
        if (initialized)
            return;

        srand(935);
        auto ssPath = std::getenv("FROST_PATH");
        if(ssPath == nullptr) {
            std::cout << "Environment variable FROST_PATH not set\n";
        }
        std::string ssPathStr = std::string(ssPath);

        char sep = '/';
        std::string matrix_path;
        if (ssPathStr[ssPathStr.length()] == sep) {
            matrix_path = ssPathStr + matrix_name;
        } else {
            matrix_path = ssPathStr + "/" + matrix_name;
        }

        B = inputCache.getTensor(matrix_path, Sparse, true);
        NUM_I = inputCache.num_i;
        NUM_J = inputCache.num_j;
        NUM_K = inputCache.num_k;

        taco::Tensor<double> C_("C", {NUM_K, NUM_L}, {taco::ModeFormat::Dense, taco::ModeFormat::Dense});
        C = C_;
        for (int k = 0; k < NUM_K; k++) {
            for (int l = 0; l < NUM_L; l++) {
                float rand_float = (float)rand()/(float)(RAND_MAX);
                C.insert({k, l}, (double) ((int) (rand_float*3)));
            }
        }

        B.pack();
        C.pack();

        std::vector<taco::IndexVar> reorder_{chunk, fpos2, kpos1, l, kpos2};
        compute_reordering(reorder_);
        // Avoid duplicate reinitialize
        initialized = true;
    }
    taco::IndexStmt schedule(int CHUNK_SIZE=16, int UNROLL_FACTOR=8, int order=0) {
        using namespace taco;
        std::vector<taco::IndexVar> reorder = get_reordering(order);
        return stmt.fuse(i, j, f)
                .pos(f, fpos, B(i,j,k))
                .split(fpos, chunk, fpos2, CHUNK_SIZE)
                .pos(k, kpos, B(i,j,k))
                .split(kpos, kpos1, kpos2, UNROLL_FACTOR)
                .reorder(reorder)
                .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
                .parallelize(kpos2, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
    }
    void generate_schedule(int chunk_size, int unroll_factor, int order) {
        A(i,j,l) = B(i,j,k) * C(k,l);

        stmt = A.getAssignment().concretize();
        stmt = schedule(chunk_size, unroll_factor, order);
    }
    void compute_cold_run() {
        A.compile(stmt);
        A.assemble();
        A.compute();
    }

    double compute_unscheduled() {
        taco::Tensor<double> result = copyNonZeroStructure({NUM_I, NUM_J, NUM_L}, {taco::Sparse, taco::Sparse, taco::Dense}, B, 2);
        result(i,j,l) = B(i,j,k) * C(k,l);
        taco::util::Timer timer;
        result.setPreserveNonZero(true);
        result.setAssembleWhileCompute(false);
        result.setNeedsAssemble(false);
        result.compile();
//        result.assemble();
        timer.start();
        result.compute();
        timer.stop();
        return timer.getResult().mean;
    }

    taco::IndexStmt schedule(taco::IndexStmt &sched, std::vector<int> order, int chunk_size=16, int unroll_factor=8, int omp_scheduling_type=0, int omp_chunk_size=0, int omp_num_threads=32) {
        using namespace taco;
        // std::vector<taco::IndexVar> reorder = get_reordering(order);
        std::vector<taco::IndexVar> reorder; //= get_reordering(order);
        taco::taco_set_num_threads(omp_num_threads);
        reorder.reserve(order.size());
        if(omp_scheduling_type == 0) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Static, omp_chunk_size);
        }
        else if(omp_scheduling_type == 1) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Dynamic, omp_chunk_size);
        }
        get_reordering(reorder, order);
        return sched.fuse(i, j, f)
                .pos(f, fpos, B(i,j,k))
                .split(fpos, chunk, fpos2, chunk_size)
                .pos(k, kpos, B(i,j,k))
                .split(kpos, kpos1, kpos2, unroll_factor)
                .reorder(reorder)
                .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
                .parallelize(kpos2, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
    }

    void schedule_and_compute(taco::Tensor<double> &result_, int chunk_size, int unroll_factor, std::vector<int> order, int omp_scheduling_type=0, int omp_chunk_size=0, int num_threads=32, bool default_config=false, int num_reps=20) {
	      taco::Tensor<double> result = copyNonZeroStructure({NUM_I, NUM_J, NUM_L}, {taco::Sparse, taco::Sparse, taco::Dense}, B, 2);
        result(i,j,l) = B(i,j,k) * C(k,l);

        taco::IndexStmt sched = result.getAssignment().concretize();
        // std::vector<int> order_(0,1,2,3,4);
        sched = schedule(sched, order, chunk_size, unroll_factor, omp_scheduling_type, omp_chunk_size, num_threads);

        if(cold_run) {
            for(int i = 0; i < 5; i++) {
                // taco::Tensor<double> temp_result({NUM_I, NUM_J}, taco::dense);
                compute_cold_run();
            }
            cold_run = false;
        }


        taco::util::Timer timer;
        std::vector<double> compute_times;
        timer.clear_cache();
        result.setPreserveNonZero(true);
        result.setNeedsAssemble(false);
        result.compile(sched);

//        result.assemble();
        for(int i = 0; i < num_reps; i++) {
            timer.start();
            result.setPreserveNonZero(true);

            result.setNeedsCompute(true);
            result.compute();
            timer.stop();

            double temp_compute_time = timer.getResult().mean;

            compute_times.push_back(temp_compute_time);
            if(temp_compute_time > 10000) {
                break;
            }
        }
        compute_time = med(compute_times);


        if(default_config) {
            default_compute_time = timer.getResult().mean;
        }
        timer.clear_cache();
    }

    void compute(bool default_config = false) override
    {
        // if(cold_run) {
            // compute_cold_run();
            // cold_run = false;
        // }
        taco::util::Timer timer;

        A(i,j,l) = B(i,j,k) * C(k,l);
        A.compile(stmt);
        A.assemble();
        timer.start();
        A.compute();
        timer.stop();

        compute_time = timer.getResult().mean;
        if (default_config)
        {
            default_compute_time = timer.getResult().mean;
        }
    }

};

class MTTKRP : public tacoOp {
public:
    int NUM_I;
    int NUM_J;
    int NUM_K;
    int NUM_L;
    int NUM_M;
    float SPARSITY;
    bool initialized;
    bool cold_run;
    taco::Tensor<double> A;
    taco::Tensor<double> B;
    taco::Tensor<double> C;
    taco::Tensor<double> D;
    taco::Tensor<double> E;
    taco::IndexStmt stmt;
    taco::IndexVar i, i1, i2, k, l, m, j;
    int run_mode;
    MTTKRP() : run_mode(1), initialized{false}, cold_run{true},
            i("i"), i1("i1"), i2("i2"), k("k"), l("l"), m("m"), j("j"){}
    MTTKRP(int mode, int NUM_I = 1000, int NUM_J = 1000, int NUM_K = 1000, int NUM_L = 1000, int NUM_M = 1000, float SPARSITY = .1) : NUM_I{NUM_I},
                                                                                     NUM_J{NUM_J},
                                                                                     NUM_K{NUM_K},
                                                                                     NUM_L{NUM_L},
                                                                                     SPARSITY{SPARSITY},
                                                                                     initialized{false},
                                                                                     cold_run{true},
                                                                                     A("A", {NUM_I, NUM_J}, {taco::Dense, taco::Dense}),
                                                                                     B("B", {NUM_I, NUM_K, NUM_L, NUM_M}, {taco::Dense, taco::Sparse, taco::Sparse, taco::Sparse}),
                                                                                     C("C", {NUM_K, NUM_J}, {taco::Dense, taco::Dense}),
                                                                                     D("D", {NUM_L, NUM_J}, {taco::Dense, taco::Dense}),
                                                                                     E("E", {NUM_M, NUM_J}, {taco::Dense, taco::Dense}),
                                                                                     i("i"), i1("i1"), i2("i2"), k("k"), l("l"), m("m"), j("j")
    {
    }
    void set_cold_run() { cold_run = true; }
    void initialize_data(int mode = RANDOM) override
    {
        //TODO: Implement read from matrix market mode
        using namespace taco;

        // Sanity check: avoid initializing multiple times
        if (initialized)
            return;

        srand(935);
        auto ssPath = std::getenv("FROST_PATH");
        if(ssPath == nullptr) {
            std::cout << "Environment variable FROST_PATH not set\n";
        }
        std::string ssPathStr = std::string(ssPath);

        char sep = '/';
        std::string matrix_path;
        if (ssPathStr[ssPathStr.length()] == sep) {
            matrix_path = ssPathStr + matrix_name;
        } else {
            matrix_path = ssPathStr + "/" + matrix_name;
        }

        B = inputCache.getTensor(matrix_path, Sparse, true);
        // B(i,k,l,m)
        NUM_I = B.getDimensions()[0];
        NUM_K = B.getDimensions()[1];
        NUM_L = B.getDimensions()[2];
        NUM_M = B.getDimensions()[3];
        // NUM_J = 32;

        std::cout << "Size: " << NUM_I << ", " << NUM_K << ", " << NUM_L << ", " << NUM_M << ", " << NUM_J << std::endl;
        // NUM_I = inputCache.num_i;
        // NUM_J = inputCache.num_j;
        // NUM_K = inputCache.num_k;

        taco::Tensor<double> C_("C", {NUM_K, NUM_J}, {taco::ModeFormat::Dense, taco::ModeFormat::Dense});
        taco::Tensor<double> D_("D", {NUM_L, NUM_J}, {taco::ModeFormat::Dense, taco::ModeFormat::Dense});
        taco::Tensor<double> E_("E", {NUM_M, NUM_J}, {taco::ModeFormat::Dense, taco::ModeFormat::Dense});
        // C(k,j) * D(l,j) * E(m,j)
        C = C_;
        D = D_;
        E = E_;
        for (int k = 0; k < NUM_K; k++) {
            for (int j = 0; j < NUM_J; j++) {
                float rand_float = (float)rand()/(float)(RAND_MAX);
                C.insert({k, j}, (double) ((int) (rand_float*3)));
            }
        }
        for (int l = 0; l < NUM_L; l++) {
            for (int j = 0; j < NUM_J; j++) {
                float rand_float = (float)rand()/(float)(RAND_MAX);
                D.insert({l, j}, (double) ((int) (rand_float*3)));
            }
        }
        for (int m = 0; m < NUM_M; m++) {
            for (int j = 0; j < NUM_J; j++) {
                float rand_float = (float)rand()/(float)(RAND_MAX);
                E.insert({m, j}, (double) ((int) (rand_float*3)));
            }
        }

        B.pack();
        C.pack();
        D.pack();
        E.pack();

        std::vector<taco::IndexVar> reorder_{i1, i2, k, l, m, j};
        compute_reordering(reorder_);
        // Avoid duplicate reinitialize
        initialized = true;
    }
    // taco::IndexStmt schedule(int CHUNK_SIZE=16, int UNROLL_FACTOR=8, int order=0) {
    //     using namespace taco;
    //     std::vector<taco::IndexVar> reorder = get_reordering(order);
    //     return stmt.fuse(i, j, f)
    //             .pos(f, fpos, B(i,j,k))
    //             .split(fpos, chunk, fpos2, CHUNK_SIZE)
    //             .pos(k, kpos, B(i,j,k))
    //             .split(kpos, kpos1, kpos2, UNROLL_FACTOR)
    //             .reorder(reorder)
    //             .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
    //             .parallelize(kpos2, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
    // }
    void generate_schedule(int chunk_size, int unroll_factor, int order) {
        // A(i,j,l) = B(i,j,k) * C(k,l);
        A(i,j) = B(i,k,l,m) * C(k,j) * D(l,j) * E(m,j);

        stmt = A.getAssignment().concretize();
        // stmt = schedule(chunk_size, unroll_factor, order);
    }
    void compute_cold_run() {
        A.compile(stmt);
        A.assemble();
        A.compute();
    }

    double compute_unscheduled() {
        taco::Tensor<double> result({NUM_I, NUM_J}, taco::dense);

        std::cout << "Inside compute unsched: " << NUM_J << std::endl;
        // result(i,j,l) = B(i,j,k) * C(k,l);
        result(i,j) = B(i,k,l,m) * C(k,j) * D(l,j) * E(m,j);

        // std::cout << B << std::endl;
        taco::util::Timer timer;
        result.compile();
        result.assemble();
        timer.start();
        result.compute();
        timer.stop();
        return timer.getResult().mean;
    }

    taco::IndexStmt schedule(taco::IndexStmt &sched, std::vector<int> order, int chunk_size=16, int unroll_factor=8, int omp_scheduling_type=0, int omp_chunk_size=0, int omp_num_threads=32) {
        using namespace taco;
        // std::vector<taco::IndexVar> reorder = get_reordering(order);
        std::vector<taco::IndexVar> reorder; //= get_reordering(order);
        taco::taco_set_num_threads(omp_num_threads);
        reorder.reserve(order.size());
        if(omp_scheduling_type == 0) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Static, omp_chunk_size);
        }
        else if(omp_scheduling_type == 1) {
            taco::taco_set_parallel_schedule(taco::ParallelSchedule::Dynamic, omp_chunk_size);
        }
        get_reordering(reorder, order);
        return sched.split(i, i1, i2, chunk_size)
            .reorder({i1, i2, k, l, m, j})
            .parallelize(i1, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
        // return sched.split(i, j, f)
        //         .pos(f, fpos, B(i,j,k))
        //         .split(fpos, chunk, fpos2, chunk_size)
        //         .pos(k, kpos, B(i,j,k))
        //         .split(kpos, kpos1, kpos2, unroll_factor)
        //         .reorder(reorder)
        //         .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
        //         .parallelize(kpos2, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
    }

    void schedule_and_compute(taco::Tensor<double> &result_, int chunk_size, int unroll_factor, std::vector<int> order, int omp_scheduling_type=0, int omp_chunk_size=0, int num_threads=32, bool default_config=false, int num_reps=20) {
        taco::Tensor<double> result("result", {NUM_I, NUM_J}, taco::dense);
        // result(i,j,l) = B(i,j,k) * C(k,l);
        // result(i,j) = B(i,k,l) * C(k,j) * D(l,j);

        // std::cout << "Dim: " << result.getDimensions()[1] << std::endl;
        // std::cout << C.getDimensions()[1] << std::endl;
        // std::cout << D.getDimensions()[1] << std::endl;
        // std::cout << E.getDimensions()[1] << std::endl;
        result(i,j) = B(i,k,l,m) * C(k,j) * D(l,j) * E(m,j);

        taco::IndexStmt sched = result.getAssignment().concretize();
        // std::vector<int> order_(0,1,2,3,4);
        sched = schedule(sched, order, chunk_size, unroll_factor, omp_scheduling_type, omp_chunk_size, num_threads);

        std::cout<<sched<<std::endl;

        if(cold_run) {
            for(int i = 0; i < 5; i++) {
                // taco::Tensor<double> temp_result({NUM_I, NUM_J}, taco::dense);
                compute_cold_run();
            }
            cold_run = false;
        }


        taco::util::Timer timer;
        std::vector<double> compute_times;
        timer.clear_cache();
        result.compile(sched);
        result.setNeedsAssemble(true);
        result.assemble();
        for(int i = 0; i < num_reps; i++) {
            timer.start();
            result.setNeedsCompute(true);
            result.compute();
            timer.stop();

            double temp_compute_time = timer.getResult().mean;
            std::cout << "Inside compute sched: " << temp_compute_time << std::endl;
        

            compute_times.push_back(temp_compute_time);
            if(temp_compute_time > 10000) {
                break;
            }
        }
        compute_time = med(compute_times);


        if(default_config) {
            default_compute_time = timer.getResult().mean;
        }
        timer.clear_cache();
    }

    void compute(bool default_config = false) override
    {
        if(cold_run) {
            compute_cold_run();
            cold_run = false;
        }
        taco::util::Timer timer;

        // A(i,j) = B(i,k,l) * C(k,j) * D(l,j);
        A(i,j) = B(i,k,l,m) * C(k,j) * D(l,j) * E(m,j);
        A.compile(stmt);
        A.assemble();
        timer.start();
        A.compute();
        timer.stop();

        compute_time = timer.getResult().mean;
        std::cout << "Inside compute sched 2: " << compute_time << std::endl;
        
        if (default_config)
        {
            default_compute_time = timer.getResult().mean;
        }
    }

};


#endif //TACO_OP_H
