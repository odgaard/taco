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

#include "hypermapper_taco_client.h"
#include "json.hpp"

using namespace std::chrono;
using json = nlohmann::json;
namespace fs = std::experimental::filesystem;
const taco::IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");


// popen2 implementation adapted from:
// https://github.com/vi/syscall_limiter/blob/master/writelimiter/popen2.c
struct popen2 {
  pid_t child_pid;
  int from_child, to_child;
};

int popen2(const char *cmdline, struct popen2 *childinfo) {
  pid_t p;
  int pipe_stdin[2], pipe_stdout[2];

  if (pipe(pipe_stdin))
    return -1;
  if (pipe(pipe_stdout))
    return -1;

  printf("pipe_stdin[0] = %d, pipe_stdin[1] = %d\n", pipe_stdin[0],
         pipe_stdin[1]);
  printf("pipe_stdout[0] = %d, pipe_stdout[1] = %d\n", pipe_stdout[0],
         pipe_stdout[1]);

  p = fork();
  if (p < 0)
    return p;   /* Fork failed */
  if (p == 0) { /* child */
    close(pipe_stdin[1]);
    dup2(pipe_stdin[0], 0);
    close(pipe_stdout[0]);
    dup2(pipe_stdout[1], 1);
    execl("/bin/sh", "sh", "-c", cmdline, 0);
    perror("execl");
    exit(99);
  }
  childinfo->child_pid = p;
  childinfo->to_child = pipe_stdin[1];
  childinfo->from_child = pipe_stdout[0];
  return 0;
}

void fatalError(const std::string &msg) {
  std::cerr << "FATAL: " << msg << std::endl;
  exit(EXIT_FAILURE);
}

// Function that creates the json scenario for hypermapper
// Arguments:
// - AppName: Name of application
// - OutputFolderName: Name of output folder
// - NumIterations: Number of HP iterations
// - NumDSERandomSamples: Number of HP random samples
// - InParams: std::vector of input parameters
// - Objectives: std::string with objective names
std::string createjson(std::string AppName, std::string OutputFoldername, int NumIterations,
                  int NumDSERandomSamples, std::vector<HMInputParamBase *> &InParams,
                  std::vector<std::string> Objectives) {

  std::string CurrentDir = fs::current_path();
  std::string OutputDir = CurrentDir + "/" + OutputFoldername + "/";
  if (fs::exists(OutputDir)) {
    std::cerr << "Output directory exists, continuing!" << std::endl;
  } else {

    std::cerr << "Output directory does not exist, creating!" << std::endl;
    if (!fs::create_directory(OutputDir)) {
      fatalError("Unable to create Directory: " + OutputDir);
    }
  }
  json HMScenario;
  HMScenario["application_name"] = AppName;
  HMScenario["optimization_objectives"] = json(Objectives);
  HMScenario["print_best"] = "auto";
  HMScenario["hypermapper_mode"]["mode"] = "client-server";
  HMScenario["run_directory"] = CurrentDir;
  HMScenario["log_file"] = OutputFoldername + "/log_" + AppName + ".log";
  HMScenario["optimization_iterations"] = NumIterations;
  HMScenario["models"]["model"] = "random_forest";

  HMScenario["output_data_file"] =
      OutputFoldername + "/" + AppName + "_output_data.csv";
  HMScenario["output_pareto_file"] =
      OutputFoldername + "/" + AppName + "_output_pareto.csv";
  HMScenario["output_image"]["output_image_pdf_file"] =
      OutputFoldername + "_" + AppName + "_output_image.pdf";

  json HMDOE;
  HMDOE["doe_type"] = "standard latin hypercube"; // "random sampling";
  HMDOE["number_of_samples"] = NumDSERandomSamples;

  HMScenario["design_of_experiment"] = HMDOE;

  for (auto InParam : InParams) {
    json HMParam;
    HMParam["parameter_type"] = getTypeAsString(InParam->getType());
    switch (InParam->getDType()) {
      case Int:
        HMParam["values"] = json(static_cast<HMInputParam<int>*>(InParam)->getRange());
        break;
      case Float:
        HMParam["values"] = json(static_cast<HMInputParam<float>*>(InParam)->getRange());
        break;
    }
    HMScenario["input_parameters"][InParam->getKey()] = HMParam;
  }

  //  std::cout << setw(4) << HMScenario << std::endl;
  ofstream HyperMapperScenarioFile;

  std::string JSonFileNameStr =
      CurrentDir + "/" + OutputFoldername + "/" + AppName + "_scenario.json";

  HyperMapperScenarioFile.open(JSonFileNameStr);
  if (HyperMapperScenarioFile.fail()) {
    fatalError("Unable to open file: " + JSonFileNameStr);
  }
  std::cout << "Writing JSON file to: " << JSonFileNameStr << std::endl;
  HyperMapperScenarioFile << setw(4) << HMScenario << std::endl;
  return JSonFileNameStr;
}

// Function that populates input parameters
int collectInputParamsSpMV(std::vector<HMInputParamBase *> &InParams) {
  int numParams = 0;

  std::vector<int> chunkSizeRange{2, 32};

  std::cout << "In here" << std::endl;
  HMInputParam<int> *chunkSizeParam = new HMInputParam<int>("chunk_size", ParamType::Integer);
  chunkSizeParam->setRange(chunkSizeRange);
  InParams.push_back(chunkSizeParam);
    // std::cout << "Param: " << *InParams[0] << "\n";
  numParams++;
  return numParams;
}

// Function that populates input parameters
int collectInputParamsSpMM(std::vector<HMInputParamBase *> &InParams) {
  int numParams = 0;

  std::vector<int> chunkSizeRange{2, 32};
  std::vector<int> unrollFactorRange{2, 32};

  HMInputParam<int> *chunkSizeParam = new HMInputParam<int>("chunk_size", ParamType::Integer);
  chunkSizeParam->setRange(chunkSizeRange);
  InParams.push_back(chunkSizeParam);
  numParams++;

  HMInputParam<int> *unrollFactorParam = new HMInputParam<int>("unroll_factor", ParamType::Integer);
  chunkSizeParam->setRange(chunkSizeRange);
  InParams.push_back(unrollFactorParam);
  numParams++;
  return numParams;
}

// Function that populates input parameters
int collectInputParamsSDDMM(std::vector<HMInputParamBase *> &InParams) {
  int numParams = 0;

  std::vector<int> chunkSizeRange{2, 32};

  HMInputParam<int> *chunkSizeParam = new HMInputParam<int>("chunk_size", ParamType::Integer);
  chunkSizeParam->setRange(chunkSizeRange);
  InParams.push_back(chunkSizeParam);
  numParams++;
  return numParams;
}

// Function that populates input parameters
int collectInputParamsTTV(std::vector<HMInputParamBase *> &InParams) {
  int numParams = 0;

  std::vector<int> chunkSizeRange{2, 32};

  HMInputParam<int> *chunkSizeParam = new HMInputParam<int>("chunk_size", ParamType::Integer);
  chunkSizeParam->setRange(chunkSizeRange);
  InParams.push_back(chunkSizeParam);
  numParams++;
  return numParams;
}

// Function that populates input parameters
int collectInputParamsTTM(std::vector<HMInputParamBase *> &InParams) {
  int numParams = 0;

  std::vector<int> chunkSizeRange{2, 32};

  HMInputParam<int> *chunkSizeParam = new HMInputParam<int>("chunk_size", ParamType::Integer);
  chunkSizeParam->setRange(chunkSizeRange);
  InParams.push_back(chunkSizeParam);
  numParams++;
  return numParams;
}

int collectInputParams(std::vector<HMInputParamBase *> &InParams, std::string test_name) {
  if (test_name == "SpMV")
    return collectInputParamsSpMV(InParams);
  if (test_name == "SpMM")
    return collectInputParamsSpMM(InParams);
  if (test_name == "SDDMM")
    return collectInputParamsSDDMM(InParams);
  if (test_name == "TTV")
    return collectInputParamsTTV(InParams);
  if (test_name == "TTM")
    return collectInputParamsTTM(InParams);
  else {
    std::cout << "Test case not implemented yet" << std::endl;
    exit(-1);
  }
}

// Free memory of input parameters
void deleteInputParams(std::vector<HMInputParamBase *> &InParams) {
  for (auto p : InParams) {
    switch(p->getDType()) {
      case Int:
        delete static_cast<HMInputParam<int>*>(p);
        break;
      case Float:
        delete static_cast<HMInputParam<float>*>(p);
        break;
      default:
        fatalError("Trying to free unhandled data type.");
    }
  }
}


// Function for mapping input parameter based on key
auto findHMParamByKey(std::vector<HMInputParamBase *> &InParams, std::string Key) {
  for (auto it = InParams.begin(); it != InParams.end(); ++it) {
    HMInputParamBase Param = **it;
    if (Param == Key) {
      return it;
    }
  }
  return InParams.end();
}

//Function that sets the input parameter value
void setInputValue(HMInputParamBase *Param, std::string ParamVal) {
  switch(Param->getDType()) {
    case Int:
      static_cast<HMInputParam<int>*>(Param)->setVal(stoi(ParamVal));
      break;
    case Float:
      static_cast<HMInputParam<float>*>(Param)->setVal(stof(ParamVal));
      break;
  }
}

//Function that performs the taco scheduling
static taco::IndexStmt scheduleSpMVCPU(taco::IndexStmt stmt, int CHUNK_SIZE=16) {
  using namespace taco;
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .reorder({i0, i1, j})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

static taco::IndexStmt scheduleSpMMCPU(taco::IndexStmt stmt, taco::Tensor<double> A, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  using namespace taco;
  IndexVar i0("i0"), i1("i1"), kbounded("kbounded"), k0("k0"), k1("k1"), jpos("jpos"), jpos0("jpos0"), jpos1("jpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(j, jpos, A(i,j))
          .split(jpos, jpos0, jpos1, UNROLL_FACTOR)
          .reorder({i0, i1, jpos0, k, jpos1})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(k, ParallelUnit::CPUVector, OutputRaceStrategy::IgnoreRaces);
}

taco::IndexStmt scheduleSDDMMCPU(taco::IndexStmt stmt, taco::Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  using namespace taco;
  IndexVar i0("i0"), i1("i1"), kpos("kpos"), kpos0("kpos0"), kpos1("kpos1");
  return stmt.split(i, i0, i1, CHUNK_SIZE)
          .pos(k, kpos, B(i,k))
          .split(kpos, kpos0, kpos1, UNROLL_FACTOR)
          .reorder({i0, i1, kpos0, j, kpos1})
          .parallelize(i0, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(kpos1, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);
}

taco::IndexStmt scheduleTTVCPU(taco::IndexStmt stmt, taco::Tensor<double> B, int CHUNK_SIZE=16) {
  using namespace taco;
  IndexVar f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2");
  return stmt.fuse(i, j, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, chunk, fpos2, CHUNK_SIZE)
          .reorder({chunk, fpos2, k})
          .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
}

taco::IndexStmt scheduleTTMCPU(taco::IndexStmt stmt, taco::Tensor<double> B, int CHUNK_SIZE=16, int UNROLL_FACTOR=8) {
  using namespace taco;
  IndexVar f("f"), fpos("fpos"), chunk("chunk"), fpos2("fpos2"), kpos("kpos"), kpos1("kpos1"), kpos2("kpos2");
  return stmt.fuse(i, j, f)
          .pos(f, fpos, B(i,j,k))
          .split(fpos, chunk, fpos2, CHUNK_SIZE)
          .pos(k, kpos, B(i,j,k))
          .split(kpos, kpos1, kpos2, UNROLL_FACTOR)
          .reorder({chunk, fpos2, kpos1, l, kpos2})
          .parallelize(chunk, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces)
          .parallelize(kpos2, ParallelUnit::CPUVector, OutputRaceStrategy::ParallelReduction);;
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveSpMVDense(std::vector<HMInputParamBase *> &InputParams) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();

  //Initialize tensors
  int NUM_I = 10000;
  int NUM_J = 10000;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> x("x", {NUM_J}, Format({Dense}));
  Tensor<double> y("y", {NUM_I}, Format({Dense}));

  //Populate tensors with random values
  srand(120);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float * 3 / SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    float rand_float = (float)rand()/(float)(RAND_MAX);
    x.insert({j}, (double) ((int) (rand_float*3/SPARSITY)));
  }

  x.pack();
  A.pack();

  //Define tensor operation (spmv)
  y(i) = A(i, j) * x(j);

  //Initiate SpMV scheduling passing in chunk_size (param to optimize)
  IndexStmt stmt = y.getAssignment().concretize();
  stmt = scheduleSpMVCPU(stmt, chunk_size);

  taco::util::Timer timer;

  y.compile(stmt);
  y.assemble();
  timer.start();
  y.compute();
  timer.stop();

  Obj.compute_time = timer.getResult().mean;
  return Obj;
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveSpMMDense(std::vector<HMInputParamBase *> &InputParams) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();
  int unroll_factor = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();

  //Initialize tensors
  int NUM_I = 10000;
  int NUM_J = 10000;
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});

  //Populate tensors with random values
  srand(120);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float * 3 / SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  A.pack();
  B.pack();

  //Define tensor operation (spmv)
  C(i, k) = A(i, j) * B(j, k);

  //Initiate SpMV scheduling passing in chunk_size (param to optimize)
  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpMMCPU(stmt, A, chunk_size, unroll_factor);

  taco::util::Timer timer;

  C.compile(stmt);
  C.assemble();
  timer.start();
  C.compute();
  timer.stop();

  Obj.compute_time = timer.getResult().mean;
  return Obj;
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveSDDMMDense(std::vector<HMInputParamBase *> &InputParams) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();

  //Initialize tensors
  int NUM_I = 10000;
  int NUM_J = 10000;
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});

  //Populate tensors with random values
  srand(120);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float * 3 / SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  A.pack();
  B.pack();

  //Define tensor operation (spmv)
  C(i, k) = A(i, j) * B(j, k);

  //Initiate SpMV scheduling passing in chunk_size (param to optimize)
  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpMMCPU(stmt, A, chunk_size);

  taco::util::Timer timer;

  C.compile(stmt);
  C.assemble();
  timer.start();
  C.compute();
  timer.stop();

  Obj.compute_time = timer.getResult().mean;
  return Obj;
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveTTVDense(std::vector<HMInputParamBase *> &InputParams) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();

  //Initialize tensors
  int NUM_I = 10000;
  int NUM_J = 10000;
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});

  //Populate tensors with random values
  srand(120);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float * 3 / SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  A.pack();
  B.pack();

  //Define tensor operation (spmv)
  C(i, k) = A(i, j) * B(j, k);

  //Initiate SpMV scheduling passing in chunk_size (param to optimize)
  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpMMCPU(stmt, A, chunk_size);

  taco::util::Timer timer;

  C.compile(stmt);
  C.assemble();
  timer.start();
  C.compute();
  timer.stop();

  Obj.compute_time = timer.getResult().mean;
  return Obj;
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveTTMDense(std::vector<HMInputParamBase *> &InputParams) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();

  //Initialize tensors
  int NUM_I = 10000;
  int NUM_J = 10000;
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<double> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<double> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<double> C("C", {NUM_I, NUM_K}, {Dense, Dense});

  //Populate tensors with random values
  srand(120);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (double) ((int) (rand_float * 3 / SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      B.insert({j, k}, (double) ((int) (rand_float*3/SPARSITY)));
    }
  }

  A.pack();
  B.pack();

  //Define tensor operation (spmv)
  C(i, k) = A(i, j) * B(j, k);

  //Initiate SpMV scheduling passing in chunk_size (param to optimize)
  IndexStmt stmt = C.getAssignment().concretize();
  stmt = scheduleSpMMCPU(stmt, A, chunk_size);

  taco::util::Timer timer;

  C.compile(stmt);
  C.assemble();
  timer.start();
  C.compute();
  timer.stop();

  Obj.compute_time = timer.getResult().mean;
  return Obj;
}

HMObjective calculateObjective(std::vector<HMInputParamBase *> &InParams, std::string test_name) {
  if (test_name == "SpMV")
    return calculateObjectiveSpMVDense(InParams);
  if (test_name == "SpMM")
    return calculateObjectiveSpMMDense(InParams);
  if (test_name == "SDDMM")
    return calculateObjectiveSDDMMDense(InParams);
  if (test_name == "TTV")
    return calculateObjectiveTTVDense(InParams);
  if (test_name == "TTM")
    return calculateObjectiveTTMDense(InParams);
  else {
    std::cout << "Test case not implemented yet" << std::endl;
    exit(-1);
  }
}

int main(int argc, char **argv) {

  if (!getenv("HYPERMAPPER_HOME")) {
    std::string ErrMsg = "Environment variables are not set!\n";
    ErrMsg += "Please set HYPERMAPPER_HOME before running this ";
    fatalError(ErrMsg);
  }

  srand(0);

  std::string test_name = "SpMM";

  // Set these values accordingly
  std::string OutputFoldername = "outdata";
  std::string AppName = "cpp_taco";
  int NumIterations = 20;
  int NumSamples = 10;
  std::vector<std::string> Objectives = {"compute_time"};

  // Create output directory if it doesn't exist
  std::string CurrentDir = fs::current_path();
  std::string OutputDir = CurrentDir + "/" + OutputFoldername + "/";
  if (fs::exists(OutputDir)) {
    std::cerr << "Output directory exists, continuing!" << std::endl;
  } else {

    std::cerr << "Output directory does not exist, creating!" << std::endl;
    if (!fs::create_directory(OutputDir)) {
      fatalError("Unable to create Directory: " + OutputDir);
    }
  }

  // Collect input parameters
  std::vector<HMInputParamBase *> InParams;

  int numParams = collectInputParams(InParams, test_name);
  std::cout << numParams << std::endl;
  for (auto param : InParams) {
    std::cout << "Param: " << *param << "\n";
  }
  return -1;

  // Create json scenario
  std::string JSonFileNameStr =
      createjson(AppName, OutputFoldername, NumIterations,
                 NumSamples, InParams, Objectives);

  // Launch HyperMapper
  std::string cmd("python3 ");
  cmd += getenv("HYPERMAPPER_HOME");
  cmd += "/scripts/hypermapper.py";
  cmd += " " + JSonFileNameStr;

  std::cout << "Executing command: " << cmd << std::endl;
  struct popen2 hypermapper;
  popen2(cmd.c_str(), &hypermapper);

  FILE *instream = fdopen(hypermapper.from_child, "r");
  FILE *outstream = fdopen(hypermapper.to_child, "w");

  const int max_buffer = 1000;
  char buffer[max_buffer];
  // Loop that communicates with HyperMapper
  // Everything is done through function calls,
  // there should be no need to modify bellow this line.
  char* fgets_res;
  int i = 0;
  while (true) {
    fgets_res = fgets(buffer, max_buffer, instream);
    if (fgets_res == NULL) {
      fatalError("'fgets' reported an error.");
    }
    std::cout << "Iteration: " << i << std::endl;
    std::cout << "Recieved: " << buffer;
    // Receiving Num Requests
    std::string bufferStr(buffer);
    if (!bufferStr.compare("End of HyperMapper\n")) {
      std::cout << "Hypermapper completed!\n";
      break;
    }
    std::string NumReqStr = bufferStr.substr(bufferStr.find(' ') + 1);
    int numRequests = stoi(NumReqStr);
    // Receiving input param names
    fgets_res = fgets(buffer, max_buffer, instream);
    if (fgets_res == NULL) {
      fatalError("'fgets' reported an error.");
    }
    bufferStr = std::string(buffer);
    std::cout << "Recieved: " << buffer;
    size_t pos = 0;
    // Create mapping for InputParam objects to keep track of order
    map<int, HMInputParamBase *> InputParamsMap;
    std::string response;
    for (int param = 0; param < numParams; param++) {
      size_t len = bufferStr.find_first_of(",\n", pos) - pos;
      std::string ParamStr = bufferStr.substr(pos, len);
      //      std::cout << "  -- param: " << ParamStr << "\n";
      auto paramIt = findHMParamByKey(InParams, ParamStr);
      if (paramIt != InParams.end()) {
        InputParamsMap[param] = *paramIt;
        response += ParamStr;
        response += ",";
      } else {
        fatalError("Unknown parameter received!");
      }
      pos = bufferStr.find_first_of(",\n", pos) + 1;
    }
    for (auto objString : Objectives)
      response += objString + ",";
    response += "\n";
    // For each request
    for (int request = 0; request < numRequests; request++) {
      // Receiving paramter values
      fgets_res = fgets(buffer, max_buffer, instream);
      if (fgets_res == NULL) {
        fatalError("'fgets' reported an error.");
      }
      std::cout << "Received: " << buffer;
      bufferStr = std::string(buffer);
      pos = 0;
      for (int param = 0; param < numParams; param++) {
        size_t len = bufferStr.find_first_of(",\n", pos) - pos;
        std::string ParamValStr = bufferStr.substr(pos, len);
        setInputValue(InputParamsMap[param], ParamValStr);
        response += ParamValStr;
        response += ",";
        pos = bufferStr.find_first_of(",\n", pos) + 1;
      }
      HMObjective Obj = calculateObjective(InParams, test_name);  // Function to run hypermapper on
      response += std::to_string(Obj.compute_time);
      response += "\n";
    }
    std::cout << "Response:\n" << response;
    fputs(response.c_str(), outstream);
    fflush(outstream);
    i++;
  }

  deleteInputParams(InParams);
  close(hypermapper.from_child);
  close(hypermapper.to_child);

  FILE *fp;
  std::string cmdPareto("python3 ");
  cmdPareto += getenv("HYPERMAPPER_HOME");
  cmdPareto += "/scripts/plot_optimization_results.py -j";
  cmdPareto += " " + JSonFileNameStr;
  cmdPareto += " -i outdata";
  std::cout << "Executing " << cmdPareto << std::endl;
  fp = popen(cmdPareto.c_str(), "r");
  while (fgets(buffer, max_buffer, fp))
    printf("%s", buffer);
  pclose(fp);

  return 0;
}
