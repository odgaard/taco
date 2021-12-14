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
// SuiteSparseTensors ssTensors;

struct UfuncInputCache {
  template<typename U>
  std::pair<taco::Tensor<double>, taco::Tensor<double>> getUfuncInput(std::string path, U format, bool countNNZ = false, bool includeThird = false) {
    // See if the paths match.
    if (this->lastPath == path) {
      // TODO (rohany): Not worrying about whether the format was the same as what was asked for.
      return std::make_pair(this->inputTensor, this->otherTensor);
    }

    // Otherwise, we missed the cache. Load in the target tensor and process it.
    std::cout << path << std::endl;
    this->lastLoaded = taco::read(path, format);
    // We assign lastPath after lastLoaded so that if taco::read throws an exception
    // then lastPath isn't updated to the new path.
    this->lastPath = path;
    this->inputTensor = lastLoaded;
    this->otherTensor = this->inputTensor;
    if (countNNZ) {
      this->nnz = 0;
      for (auto& it : taco::iterate<double>(this->inputTensor)) {
        this->nnz++;
      }
    }
    // if (includeThird) {
    //   this->thirdTensor = shiftLastMode<int64_t, int64_t>("C", this->otherTensor);
    // }
    return std::make_pair(this->inputTensor, this->otherTensor);
  }

  taco::Tensor<double> lastLoaded;
  std::string lastPath;

  taco::Tensor<double> inputTensor;
  taco::Tensor<double> otherTensor;
//   taco::Tensor<int64_t> thirdTensor;
  int nnz;
};
UfuncInputCache inputCache;

class tacoOp {
public:
    double compute_time;
    double default_compute_time;
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
        reorderings = new LoopReordering(ordering);
        reorderings->compute_permutations();
    }
    std::pair<taco::Tensor<double>, taco::Tensor<double>> load_tensor(std::string tensorPath) {
        auto suitesparsePath = std::getenv("SUITESPARSE_PATH");
        if(suitesparsePath == nullptr) {
            std::cout << "Set SUITESPARSE_PATH" << std::endl;
            exit(1);
        }
        // std::string fullpath = std::string(suitesparsePath) + "/" + tensorPath;
        return inputCache.getUfuncInput(tensorPath, taco::CSR, true);
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
    float SPARSITY;
    bool initialized;
    bool reorder_initialized;
    bool cold_run;
    taco::Tensor<double> B;
    taco::Tensor<double> c;
    taco::Tensor<double> a;
    taco::util::Timer timer;
    taco::IndexStmt stmt;
    taco::IndexVar i0, i1, i10, i11, kpos, kpos0, kpos1;
    SpMV(int NUM_I = 10000, int NUM_J = 10000, float SPARSITY = .3) : NUM_I{NUM_I},
                                                                      NUM_J{NUM_J},
                                                                      SPARSITY{SPARSITY},
                                                                      initialized{false},
                                                                      reorder_initialized{false},
                                                                      cold_run{true},
                                                                      B("B", {NUM_I, NUM_J}, taco::CSR),
                                                                      c("c", {NUM_J}, taco::Format{taco::ModeFormat::Dense}),
                                                                      a("a", {NUM_I}, taco::Format{taco::ModeFormat::Dense}),
                                                                      i0("i0"), i1("i1"), i10("i10"), i11("i11"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1") {}
    SpMV() : initialized{false},
             reorder_initialized{false},
             cold_run{true},
             i0("i0"), i1("i1"), i10("i10"), i11("i11"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1") {}
    void initialize_data(int mode = RANDOM) override
    {
        using namespace taco;
        //TODO: Implement read from matrix market mode
        if (initialized)
            return;

        if(mode == MTX) {
            ssTensors test;
            std::tie(B, c) = load_tensor(test.tensors[10]);
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

        // taco::util::Timer timer;
        // timer.start();
        // timer.stop();
        // std::cout << "Time: " << timer.getResult().mean << std::endl;

        std::vector<taco::IndexVar> reorder_{i0, i1, j};
        compute_reordering(reorder_);
        // Avoid duplicate reinitialize
        initialized = true;
    }
    taco::IndexStmt schedule(std::vector<int> order, int CHUNK_SIZE=16, int SPLIT=0, int CHUNK_SIZE2=8) {
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
        stmt = schedule(order, chunk_size, split, chunk_size2);
    }
    void generate_schedule(int chunk_size, int split, int chunk_size2, int order) {
        a(i) = B(i,j) * c(j);

        stmt = a.getAssignment().concretize();
        stmt = schedule(chunk_size, split, chunk_size2, order);
    }
    void compute_cold_run() {
        a.compile(stmt);
        a.assemble();
        a.compute();
    }
    void compute(bool default_config = false) override {
        if(cold_run) {
            compute_cold_run();
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

};

void printToCout(taco::IndexStmt stmt) {
//   using namespace taco;
//   std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(cout, ir::CodeGen::ImplementationGen);
//   ir::Stmt compute = lower(stmt, "compute", false, true);
//   codegen->compile(compute, true);
}

class SpMM : public tacoOp {
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
    taco::IndexStmt stmt;
    taco::IndexVar i0, i1, kbounded, k0, k1, jpos, jpos0, jpos1;
    SpMM(int NUM_I = 10000, int NUM_J = 10000, int NUM_K = 1000, float SPARSITY = .3) : NUM_I{NUM_I},
                                                                                        NUM_J{NUM_J},
                                                                                        NUM_K{NUM_K},
                                                                                        SPARSITY{SPARSITY},
                                                                                        initialized{false},
                                                                                        cold_run{true},
                                                                                        A("A", {NUM_I, NUM_K}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense}),
                                                                                        B("B", {NUM_I, NUM_J}, taco::CSR),
                                                                                        C("C", {NUM_J, NUM_K}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense}),
                                                                                        i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1")
    {
    }
    void initialize_data(int mode = RANDOM) override
    {
        using namespace taco;
        //TODO: Implement read from matrix market mode
        if (initialized)
            return;
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

        B.pack();
        C.pack();

        A(i, k) = B(i, j) * C(j, k);
        std::vector<taco::IndexVar> reorder_{i0, i1, jpos0, k, jpos1};
        compute_reordering(reorder_);
        // Avoid duplicate reinitialize
        initialized = true;
    }
    taco::IndexStmt schedule(int CHUNK_SIZE=16, int UNROLL_FACTOR=8, int order=0) {
        using namespace taco;
        std::vector<taco::IndexVar> reorder = get_reordering(order);
        return stmt.split(i, i0, i1, CHUNK_SIZE)
                .pos(j, jpos, B(i,j))
                .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
                .reorder(reorder)
                .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
                .parallelize(k, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces);
    }
    void generate_schedule(int chunk_size, int unroll_factor, int order) {
        // A(i, k) = B(i, j) * C(j, k);

        stmt = A.getAssignment().concretize();
        stmt = schedule(chunk_size, unroll_factor, order);
    }
    void compute_cold_run() {
        A.compile(stmt);
        A.assemble();
        A.compute();
    }
    void compute(bool default_config = false) override
    {
        if(cold_run) {
            compute_cold_run();
            cold_run = false;
        }
        taco::util::Timer timer;

        // A(i, k) = B(i, j) * C(j, k);
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
    taco::IndexVar i0, i1, kpos, kpos0, kpos1;
    SDDMM(int NUM_I = 10000, int NUM_J = 10000, int NUM_K = 1000, float SPARSITY = .3) : NUM_I{NUM_I},
                                                                                         NUM_J{NUM_J},
                                                                                         NUM_K{NUM_K},
                                                                                         SPARSITY{SPARSITY},
                                                                                         initialized{false},
                                                                                         cold_run{true},
                                                                                         A("A", {NUM_I, NUM_K}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense}),
                                                                                         B("B", {NUM_I, NUM_K}, taco::CSR),
                                                                                         C("C", {NUM_I, NUM_J}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense}),
                                                                                         D("D", {NUM_J, NUM_K}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense}),
                                                                                         i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1")
    {
    }
    void initialize_data(int mode = RANDOM) override
    {
        //TODO: Implement read from matrix market mode
        
        using namespace taco;
        // Sanity check
        if (initialized)
            return;

        srand(268238);
        for (int i = 0; i < NUM_I; i++)
        {
            for (int j = 0; j < NUM_J; j++)
            {
                float rand_float = (float)rand() / (float)(RAND_MAX);
                C.insert({i, j}, (double)((int)(rand_float * 3 / SPARSITY)));
            }
        }

        for (int i = 0; i < NUM_I; i++)
        {
            for (int k = 0; k < NUM_K; k++)
            {
                float rand_float = (float)rand() / (float)(RAND_MAX);
                if (rand_float < SPARSITY)
                {
                    B.insert({i, k}, (double)((int)(rand_float * 3 / SPARSITY)));
                }
            }
        }

        for (int j = 0; j < NUM_J; j++)
        {
            for (int k = 0; k < NUM_K; k++)
            {
                float rand_float = (float)rand() / (float)(RAND_MAX);
                D.insert({j, k}, (double)((int)(rand_float * 3 / SPARSITY)));
            }
        }

        B.pack();
        C.pack();
        D.pack();

        A(i,k) = B(i,k) * C(i,j) * D(j,k);
        // Avoid duplicate reinitialize
        initialized = true;
        std::vector<taco::IndexVar> reorder_{i0, i1, kpos0, j, kpos1};
        compute_reordering(reorder_);
    }

    void compute_cold_run() {
        A.compile(stmt);
        A.assemble();
        A.compute();
    }

    taco::IndexStmt schedule(int CHUNK_SIZE=16, int UNROLL_FACTOR=8, int order=0) {
        //TODO: Unroll factor needs to be less than the chunk size
        using namespace taco;
        std::vector<taco::IndexVar> reorder = get_reordering(order);
        return stmt.split(i, i0, i1, CHUNK_SIZE)
                .pos(k, kpos, B(i,k))
                .split(kpos, kpos0, kpos1, UNROLL_FACTOR)
                .reorder(reorder)
                .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
                .parallelize(kpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
    }

    void generate_schedule(int chunk_size, int unroll_factor, int order) {
        A(i,k) = B(i,k) * C(i,j) * D(j,k);

        stmt = A.getAssignment().concretize();
        stmt = schedule(chunk_size, unroll_factor, order);
    }

    void compute(bool default_config = false) override
    {
        if(cold_run) {
            compute_cold_run();
            cold_run = false;
        }

        taco::util::Timer timer;

        A(i,k) = B(i,k) * C(i,j) * D(j,k);
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
    taco::IndexVar f, fpos, chunk, fpos2;
    TTV(int NUM_I = 1000, int NUM_J = 1000, int NUM_K = 1000, float SPARSITY = .3) : NUM_I{NUM_I},
                                                                                     NUM_J{NUM_J},
                                                                                     NUM_K{NUM_K},
                                                                                     SPARSITY{SPARSITY},
                                                                                     initialized{false},
                                                                                     cold_run{true},
                                                                                     A("A", {NUM_I, NUM_J}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense}),
                                                                                     B("B", {NUM_I, NUM_J, NUM_K}, taco::Format{taco::ModeFormat::Sparse, taco::ModeFormat::Sparse, taco::ModeFormat::Sparse}),
                                                                                     c("c", {NUM_K}, taco::Format{taco::ModeFormat::Dense}),
                                                                                     f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2")
    {
    }
    void initialize_data(int mode = RANDOM) override
    {
        //TODO: Implement read from matrix market mode
        using namespace taco;

        // Sanity check: avoid initializing multiple times
        if (initialized)
            return;

        srand(9536);
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
                    }
                }
            }
        }

        for (int k = 0; k < NUM_K; k++)
        {
            float rand_float = (float)rand() / (float)(RAND_MAX);
            c.insert({k}, (double)((int)(rand_float * 3)));
        }

        B.pack();
        c.pack();

        std::vector<taco::IndexVar> reorder_{chunk, fpos2, k};
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
    void compute_cold_run() {
        A.compile(stmt);
        A.assemble();
        A.compute();
    }
    void compute(bool default_config = false) override
    {
        if(cold_run) {
            compute_cold_run();
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
    TTM(int NUM_I = 1000, int NUM_J = 1000, int NUM_K = 1000, int NUM_L = 1000, float SPARSITY = .1) : NUM_I{NUM_I},
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
    void initialize_data(int mode = RANDOM) override
    {
        //TODO: Implement read from matrix market mode
        using namespace taco;

        // Sanity check: avoid initializing multiple times
        if (initialized)
            return;

        srand(935);
        for (int i = 0; i < NUM_I; i++) {
            for (int j = 0; j < NUM_J; j++) {
            for (int k = 0; k < NUM_K; k++) {
                float rand_float = (float) rand() / (float) (RAND_MAX);
                if (rand_float < SPARSITY) {
                B.insert({i, j, k}, (double) ((int) (rand_float * 3 / SPARSITY)));
                }
            }
            }
        }

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

#endif //TACO_OP_H