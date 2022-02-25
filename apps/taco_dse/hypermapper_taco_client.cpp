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

#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/transformations.h"
#include "taco/lower/lower.h"

#include "hypermapper_taco_client.h"
#include "taco_helper.h"
#include "json.hpp"

SpMV *spmv_handler;
SpMV *spmv_sparse_handler;
SpMM *spmm_handler;
SDDMM *sddmm_handler;
TTV *ttv_handler;
TTM *ttm_handler;
bool initialized = false;

using namespace std::chrono;
using json = nlohmann::json;
namespace fs = std::experimental::filesystem;
// const taco::IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
int WARP_SIZE = 32;
float default_config_time = 0.0f;
float no_sched_time = 0.0f;
int num_loops = 0;
float sparsity = 0.0f;
int num_j = 0;
std::string op;

struct popen2 {
  pid_t child_pid;
  int from_child, to_child;
};

int popen2(const char *cmdline, struct popen2 *childinfo);
std::string createjson(std::string AppName, std::string OutputFoldername, int NumIterations,
                  int NumDSERandomSamples, std::vector<HMInputParamBase *> &InParams,
                  std::vector<std::string> Objectives, std::string taco_op, std::string optimization, std::string count);
void fatalError(const std::string &msg);
int collectInputParamsSpMV(std::vector<HMInputParamBase *> &InParams, int SPLIT);
int collectInputParamsSpMM(std::vector<HMInputParamBase *> &InParams);
int collectInputParamsSDDMM(std::vector<HMInputParamBase *> &InParams);
int collectInputParamsTTV(std::vector<HMInputParamBase *> &InParams);
int collectInputParamsTTM(std::vector<HMInputParamBase *> &InParams);
int collectInputParams(std::vector<HMInputParamBase *> &InParams, std::string test_name);
void deleteInputParams(std::vector<HMInputParamBase *> &InParams);
auto findHMParamByKey(std::vector<HMInputParamBase *> &InParams, std::string Key);
void setInputValue(HMInputParamBase *Param, std::string ParamVal);
taco::IndexStmt scheduleTTMCPU(taco::IndexStmt stmt, taco::Tensor<double> B, int CHUNK_SIZE, int UNROLL_FACTOR, int order);
HMObjective calculateObjectiveSpMVDense(std::vector<HMInputParamBase *> &InputParams, std::ofstream &logger);
HMObjective calculateObjectiveSpMVSparse(std::vector<HMInputParamBase *> &InputParams, std::ofstream &logger);
HMObjective calculateObjectiveSpMMDense(std::vector<HMInputParamBase *> &InputParams, std::ofstream &logger);
HMObjective calculateObjectiveSDDMMDense(std::vector<HMInputParamBase *> &InputParams, std::ofstream &logger);
HMObjective calculateObjectiveTTVDense(std::vector<HMInputParamBase *> &InputParams, std::ofstream &logger);
HMObjective calculateObjectiveTTMDense(std::vector<HMInputParamBase *> &InputParams, std::ofstream &logger);
HMObjective calculateObjective(std::vector<HMInputParamBase *> &InParams, std::string test_name, std::ofstream &logger);


// popen2 implementation adapted from:
// https://github.com/vi/syscall_limiter/blob/master/writelimiter/popen2.c
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
                  std::vector<std::string> Objectives, std::string taco_op, std::string optimization, std::string count="0") {

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
  HMScenario["log_file"] = OutputFoldername + "/log_" + AppName + "_" + taco_op + ".log";
  if(optimization != "random_sampling") { 
    HMScenario["optimization_method"] = optimization;
    HMScenario["optimization_iterations"] = NumIterations;
  }
  else {
    HMScenario["optimization_iterations"] = 0;
  }

  // HMScenario["optimization_method"] = "local_search";
  // HMScenario["models"]["model"] = "random_forest";
  HMScenario["models"]["model"] = "gaussian_process";

  HMScenario["output_data_file"] =
      OutputFoldername + "/" + AppName + "_" + taco_op + "_" + optimization + count + "_output_data.csv";
  HMScenario["output_pareto_file"] =
      OutputFoldername + "/" + AppName + "_" + taco_op + "_output_pareto.csv";
  HMScenario["output_image"]["output_image_pdf_file"] =
      OutputFoldername + "_" + AppName + "_" + taco_op + "_output_image.pdf";

  json HMDOE;
  HMDOE["doe_type"] = "random sampling";
  HMDOE["number_of_samples"] = NumDSERandomSamples;
  if(optimization == "random_sampling") {
    HMDOE["number_of_samples"] = NumDSERandomSamples + NumIterations;
  }

  HMScenario["design_of_experiment"] = HMDOE;

  for (auto InParam : InParams) {
    json HMParam;
    HMParam["parameter_type"] = getTypeAsString(InParam->getType());
    switch (InParam->getDType()) {
      case Int:
        HMParam["values"] = json(static_cast<HMInputParam<int>*>(InParam)->getRange());
        if(InParam->getName() == "chunk_size2") {
          std::vector<std::string> constraint;
          constraint.push_back("chunk_size % chunk_size2 == 0");
          HMParam["constraints"] = json(constraint);
          std::vector<std::string> dependency;
          dependency.push_back("chunk_size");
          HMParam["dependencies"] = json(dependency);
        }
        else if(InParam->getName() == "chunk_size") {
          std::vector<std::string> constraint;
          // constraint.push_back("(chunk_size & (chunk_size - 1)) == 0");
          constraint.push_back("chunk_size % 2 == 0");
          HMParam["constraints"] = json(constraint);
          // std::vector<std::string> dependency;
          // dependency.push_back("chunk_size");
          // HMParam["dependencies"] = json(dependency);
          HMParam["parameter_default"] = 16;
        }
        else if(InParam->getName() == "reordering") {
          HMParam["parameter_default"] = 0;
        }
        else if(InParam->getName() == "unroll_factor") {
          HMParam["parameter_default"] = 8;
          std::vector<std::string> constraint;
          constraint.push_back("unroll_factor < chunk_size");
          constraint.push_back("unroll_factor % 2 == 0");
          HMParam["constraints"] = json(constraint);
          std::vector<std::string> dependency;
          dependency.push_back("chunk_size");
          HMParam["dependencies"] = json(dependency);
        }
        else if(InParam->getName() == "split") {
          HMParam["parameter_default"] = 0;
        }
        else if(InParam->getName().find("loop") != std::string::npos) {
            std::string param = InParam->getName();
            std::vector<std::string> constraint;
            std::vector<std::string> dependency;
            int last_idx = std::stoi(std::string(1, InParam->getName().back()));
            if(last_idx != 0) {
              // int i = 0;
              // for(int i = 0; i < num_loops; i++) {
                int current_idx = std::stoi(std::string(1, param.back()));
                // int i = current_idx - 1;
                std::vector<int> consts;
                // consts.push_back(i);
                for(int i = current_idx - 1; i >= 0; i--) {
                  consts.push_back(i);
                }
                for(auto idx : consts) {
                  constraint.push_back(param + " != loop" + std::to_string(idx));
                  dependency.push_back("loop" + std::to_string(idx));
                }
                // if(std::stoi(std::string(1, param.back())) != i) {
                // }
              // }
              HMParam["constraints"] = json(constraint);
              HMParam["dependencies"] = json(dependency);
            }
            else {
              // constraint.push_back(param + " > -1");
              // HMParam["constraints"] = json(constraint);
            }
            HMParam["parameter_default"] = last_idx;
        }
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
      CurrentDir + "/" + OutputFoldername + "/" + AppName + "_" + taco_op + "_" + optimization + "_scenario.json";

  HyperMapperScenarioFile.open(JSonFileNameStr);
  if (HyperMapperScenarioFile.fail()) {
    fatalError("Unable to open file: " + JSonFileNameStr);
  }
  std::cout << "Writing JSON file to: " << JSonFileNameStr << std::endl;
  HyperMapperScenarioFile << setw(4) << HMScenario << std::endl;
  return JSonFileNameStr;
}

// Function that populates input parameters
int collectInputParamsSpMV(std::vector<HMInputParamBase *> &InParams, int SPLIT=0) {
  int numParams = 0;

  std::vector<int> chunkSizeRange{2, 512};

  HMInputParam<int> *chunkSizeParam = new HMInputParam<int>("chunk_size", ParamType::Integer);
  chunkSizeParam->setRange(chunkSizeRange);
  InParams.push_back(chunkSizeParam);
  numParams++;

  int reorder_size = 3;

  if(SPLIT) {
    std::vector<int> splitRange{0, 1};
    HMInputParam<int> *splitParam = new HMInputParam<int>("split", ParamType::Categorical);
    splitParam->setRange(splitRange);
    InParams.push_back(splitParam);
    numParams++;

    std::vector<int> chunkSize2Range{2, 512};
    HMInputParam<int> *chunkSize2Param = new HMInputParam<int>("chunk_size2", ParamType::Integer);
    chunkSize2Param->setRange(chunkSize2Range);
    InParams.push_back(chunkSize2Param);
    numParams++;

    reorder_size++;
  }

  // int num_reorderings = factorial(reorder_size) - 1;
  std::vector<int> reorderRange{0, reorder_size - 1};
  for(int i = 0; i < reorder_size; i++) {
    HMInputParam<int> *reorderParam = new HMInputParam<int>("loop" + std::to_string(i), ParamType::Integer);
    reorderParam->setRange(reorderRange);
    InParams.push_back(reorderParam);
    numParams++;
  }

  num_loops = reorder_size;

  return numParams;
}

// Function that populates input parameters
int collectInputParamsSpMM(std::vector<HMInputParamBase *> &InParams) {
  int numParams = 0;

  // std::vector<int> chunkSizeRange{1, 1024};
  // std::vector<int> unrollFactorRange{1, 1024};
  std::vector<int> chunkSizeValues{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
  std::vector<int> unrollFactorValues{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

  HMInputParam<int> *chunkSizeParam = new HMInputParam<int>("chunk_size", ParamType::Ordinal);
  chunkSizeParam->setRange(chunkSizeValues);
  InParams.push_back(chunkSizeParam);
  numParams++;

  HMInputParam<int> *unrollFactorParam = new HMInputParam<int>("unroll_factor", ParamType::Ordinal);
  unrollFactorParam->setRange(unrollFactorValues);
  InParams.push_back(unrollFactorParam);
  numParams++;

  int reorder_size = 5;
  // int num_reorderings = factorial(reorder_size) - 1;
  // std::vector<int> reorderRange{0, num_reorderings};
  // HMInputParam<int> *reorderParam = new HMInputParam<int>("reordering", ParamType::Integer);
  // reorderParam->setRange(reorderRange);
  // InParams.push_back(reorderParam);
  // numParams++;

  std::vector<int> reorderRange{0, reorder_size - 1};
  for(int i = 0; i < reorder_size; i++) {
    HMInputParam<int> *reorderParam = new HMInputParam<int>("loop" + std::to_string(i), ParamType::Integer);
    reorderParam->setRange(reorderRange);
    InParams.push_back(reorderParam);
    numParams++;
  }

  num_loops = reorder_size;

  return numParams;
}

// Function that populates input parameters
int collectInputParamsSDDMM(std::vector<HMInputParamBase *> &InParams) {
  int numParams = 0;

  std::vector<int> chunkSizeRange{2, 512};
  std::vector<int> unrollFactorRange{2, 256};

  HMInputParam<int> *chunkSizeParam = new HMInputParam<int>("chunk_size", ParamType::Integer);
  chunkSizeParam->setRange(chunkSizeRange);
  InParams.push_back(chunkSizeParam);
  numParams++;

  HMInputParam<int> *unrollFactorParam = new HMInputParam<int>("unroll_factor", ParamType::Integer);
  unrollFactorParam->setRange(unrollFactorRange);
  InParams.push_back(unrollFactorParam);
  numParams++;

  int reorder_size = 5;

  std::vector<int> reorderRange{0, reorder_size - 1};
  for(int i = 0; i < reorder_size; i++) {
    HMInputParam<int> *reorderParam = new HMInputParam<int>("loop" + std::to_string(i), ParamType::Integer);
    reorderParam->setRange(reorderRange);
    InParams.push_back(reorderParam);
    numParams++;
  }

  num_loops = reorder_size;

  // int reorder_size = 5;
  // int num_reorderings = factorial(reorder_size) - 1;
  // std::vector<int> reorderRange{0, num_reorderings};
  // HMInputParam<int> *reorderParam = new HMInputParam<int>("reordering", ParamType::Integer);
  // reorderParam->setRange(reorderRange);
  // InParams.push_back(reorderParam);
  // numParams++;

  return numParams;
}

// Function that populates input parameters
int collectInputParamsTTV(std::vector<HMInputParamBase *> &InParams) {
  int numParams = 0;

  std::vector<int> chunkSizeRange{2, 512};

  HMInputParam<int> *chunkSizeParam = new HMInputParam<int>("chunk_size", ParamType::Integer);
  chunkSizeParam->setRange(chunkSizeRange);
  InParams.push_back(chunkSizeParam);
  numParams++;

  int reorder_size = 3;
  int num_reorderings = factorial(reorder_size) - 1;
  std::vector<int> reorderRange{0, num_reorderings};
  HMInputParam<int> *reorderParam = new HMInputParam<int>("reordering", ParamType::Integer);
  reorderParam->setRange(reorderRange);
  InParams.push_back(reorderParam);
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

  int reorder_size = 5;
  int num_reorderings = factorial(reorder_size) - 1;
  std::vector<int> reorderRange{0, num_reorderings};
  HMInputParam<int> *reorderParam = new HMInputParam<int>("reordering", ParamType::Integer);
  reorderParam->setRange(reorderRange);
  InParams.push_back(reorderParam);
  numParams++;

  return numParams;
}

int collectInputParams(std::vector<HMInputParamBase *> &InParams, std::string test_name) {
  if (test_name == "SpMV")
    return collectInputParamsSpMV(InParams);
  if (test_name == "SpMVSparse")
    return collectInputParamsSpMV(InParams, 1);
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


// Function that takes input parameters and generates objective
HMObjective calculateObjectiveSpMVDense(std::vector<HMInputParamBase *> &InputParams, std::ofstream &logger) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();
  // int order = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();

  std::vector<int> default_ordering;
  std::vector<int> loop_ordering;
  for(int i = 0; i < num_loops; i++) {
    // std::cout << "in here\n";
    int order = static_cast<HMInputParam<int>*>(InputParams[i + 1])->getVal();
    loop_ordering.push_back(order);
    default_ordering.push_back(i);
  }

  int NUM_I = 10000;
  int NUM_J = 10000;
  if(!initialized) {
    // spmv_handler = new SpMV(NUM_I, NUM_J);
    spmv_handler = new SpMV();
    spmv_handler->initialize_data(1);
    initialized = true;
  }

  spmv_handler->generate_schedule(chunk_size, 0, 0, loop_ordering);

  bool default_config = (chunk_size == 16 && loop_ordering == default_ordering);
  spmv_handler->compute(default_config);

  Obj.compute_time = spmv_handler->get_compute_time();

  if(chunk_size == 16 && loop_ordering == default_ordering) {
    default_config_time = spmv_handler->get_default_compute_time();
  }
  return Obj;
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveSpMVSparse(std::vector<HMInputParamBase *> &InputParams, std::ofstream &logger) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();
  int split = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();
  int chunk_size2 = static_cast<HMInputParam<int>*>(InputParams[2])->getVal();
  int order = static_cast<HMInputParam<int>*>(InputParams[3])->getVal();

  // if(no_sched_time == 0.0f)
  //   SpMVNoSched(logger);
  int NUM_I = 50000;
  int NUM_J = 50000;

  //Initialize tensors
  if(!initialized) {
    spmv_sparse_handler = new SpMV(NUM_I, NUM_J);
    spmv_sparse_handler->initialize_data();
    initialized = true;
  }

  //Initiate SpMV scheduling passing in chunk_size (param to optimize)
  spmv_sparse_handler->generate_schedule(chunk_size, split, chunk_size2, order);

  bool default_config = (chunk_size == 16 && split == 0 && order == 0);
  spmv_sparse_handler->compute(default_config);

  Obj.compute_time = spmv_sparse_handler->get_compute_time();

  if(chunk_size == 16 && order == 0 && split == 0) {
    default_config_time = spmv_sparse_handler->get_default_compute_time();
  }

  return Obj;
}

double median(vector<double> vec)
{
        typedef vector<int>::size_type vec_sz;

        vec_sz size = vec.size();
        if (size == 0)
                throw domain_error("median of an empty vector");

        sort(vec.begin(), vec.end());

        vec_sz mid = size/2;

        return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveSpMMDense(std::vector<HMInputParamBase *> &InputParams, std::ofstream &logger) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();
  int unroll_factor = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();
  // int order = static_cast<HMInputParam<int>*>(InputParams[2])->getVal();

  std::vector<int> default_ordering;
  std::vector<int> loop_ordering;
  for(int i = 0; i < num_loops; i++) {
    // std::cout << "in here\n";
    int order = static_cast<HMInputParam<int>*>(InputParams[i + 2])->getVal();
    loop_ordering.push_back(order);
    default_ordering.push_back(i);
  }

  int NUM_I = 67173;
  int NUM_J = 67173;
  int NUM_K = 1000;
  float _sparsity = .982356;

  if(!initialized) {
    // spmm_handler = new SpMM(NUM_I, NUM_J, NUM_K, sparsity);
    // spmm_handler = new SpMM(0, NUM_I, NUM_J, NUM_K, _sparsity);
    spmm_handler = new SpMM();
    // spmm_handler->initialize_data();
    spmm_handler->initialize_data(1);
    initialized = true;
    sparsity = spmm_handler->get_sparsity();
    num_j = spmm_handler->get_num_j();
    op = "SpMM";
  }


  //Initiate scheduling passing in chunk_size (param to optimize)
  // spmm_handler->generate_schedule(chunk_size, unroll_factor, order);
  spmm_handler->generate_schedule(chunk_size, unroll_factor, loop_ordering);

  bool default_config = (chunk_size == 16 && unroll_factor == 8 && loop_ordering == default_ordering);
  int num_reps = 10;
  double total_time = 0.0;
  std::vector<double> compute_times;
  for(int i = 0; i < num_reps; i++) {
    spmm_handler->compute(default_config);
    total_time = spmm_handler->get_compute_time();
    compute_times.push_back(total_time);
  }

  // double compute_time = total_time / num_reps;
  double compute_time = median(compute_times);

  Obj.compute_time = compute_time;

  if(default_config) {
    default_config_time = compute_time;
  }
  return Obj;
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveSDDMMDense(std::vector<HMInputParamBase *> &InputParams, std::ofstream &logger) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();
  int unroll_factor = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();
  // int order = static_cast<HMInputParam<int>*>(InputParams[2])->getVal();

  std::vector<int> default_ordering;
  std::vector<int> loop_ordering;
  for(int i = 0; i < num_loops; i++) {
    int order = static_cast<HMInputParam<int>*>(InputParams[i + 2])->getVal();
    loop_ordering.push_back(order);
    default_ordering.push_back(i);
  }

  //Initialize tensors
  int NUM_I = 10000;
  int NUM_J = 10000;
  int NUM_K = 1000;

  if(!initialized) {
    // sddmm_handler = new SDDMM(NUM_I, NUM_J, NUM_K);
    sddmm_handler = new SDDMM();
    sddmm_handler->initialize_data(1);
    initialized = true;
  }

  //Initiate scheduling passing in chunk_size (param to optimize)
  // sddmm_handler->generate_schedule(chunk_size, unroll_factor, order);
  sddmm_handler->generate_schedule(chunk_size, unroll_factor, loop_ordering);

  // bool default_config = (chunk_size == 16 && unroll_factor == 8 && order == 0);
  bool default_config = (chunk_size == 16 && unroll_factor == 8 && loop_ordering == default_ordering);
  int num_reps = 10;
  double total_time = 0.0;
  for(int i = 0; i < num_reps; i++) {
    sddmm_handler->compute(default_config);
    total_time += spmm_handler->get_compute_time();
  }

  double compute_time = total_time / num_reps;

  if(default_config) {
    default_config_time = compute_time;
  }
  return Obj;
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveTTVDense(std::vector<HMInputParamBase *> &InputParams, std::ofstream &logger) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();
  int order = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();

  int NUM_I = 10000;
  int NUM_J = 10000;
  int NUM_K = 1000;

  if(!initialized) {
    ttv_handler = new TTV(NUM_I, NUM_J, NUM_K);
    ttv_handler->initialize_data();
    initialized = true;
  }

  //Initiate scheduling passing in chunk_size (param to optimize)
  ttv_handler->generate_schedule(chunk_size, order);

  bool default_config = (chunk_size == 16 && order == 0);
  ttv_handler->compute(default_config);

  Obj.compute_time = ttv_handler->get_compute_time();

  if(default_config) {
    default_config_time = ttv_handler->get_default_compute_time();
  }

  return Obj;
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveTTMDense(std::vector<HMInputParamBase *> &InputParams, std::ofstream &logger) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();
  int unroll_factor = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();
  int order = static_cast<HMInputParam<int>*>(InputParams[2])->getVal();

  int NUM_I = 10000;
  int NUM_J = 10000;
  int NUM_K = 1000;

  if(!initialized) {
    ttm_handler = new TTM(NUM_I, NUM_J, NUM_K);
    ttm_handler->initialize_data();
    initialized = true;
  }

  //Initiate scheduling passing in chunk_size (param to optimize)
  ttm_handler->generate_schedule(chunk_size, unroll_factor, order);

  bool default_config = (chunk_size == 16 && unroll_factor == 8 && order == 0);
  ttm_handler->compute(default_config);

  Obj.compute_time = ttm_handler->get_compute_time();

  if(default_config) {
    default_config_time = ttm_handler->get_default_compute_time();
  }

  return Obj;
}

HMObjective calculateObjective(std::vector<HMInputParamBase *> &InParams, std::string test_name, std::ofstream &logger) {
  if (test_name == "SpMV")
    return calculateObjectiveSpMVDense(InParams, logger);
  if (test_name == "SpMVSparse")
    return calculateObjectiveSpMVSparse(InParams, logger);
  if (test_name == "SpMM")
    return calculateObjectiveSpMMDense(InParams, logger);
  if (test_name == "SDDMM")
    return calculateObjectiveSDDMMDense(InParams, logger);
  if (test_name == "TTV")
    return calculateObjectiveTTVDense(InParams, logger);
  if (test_name == "TTM")
    return calculateObjectiveTTMDense(InParams, logger);
  else {
    std::cout << "Test case not implemented yet" << std::endl;
    exit(-1);
  }
}

int main(int argc, char **argv) {

    setenv("HYPERMAPPER_HOME", "/home/ubuntu/workspace/hypermapper_dev", true);
    printf("Setting HM variable\n");
  if (!getenv("HYPERMAPPER_HOME")) {
    std::string ErrMsg = "Environment variables are not set!\n";
    ErrMsg += "Please set HYPERMAPPER_HOME before running this ";
    // fatalError(ErrMsg);
  }

  srand(0);

  // std::string test_name = "SpMM";
  std::string test_name, optimization;
  if (argv[1] == nullptr)
    test_name = "SpMV";
  else 
    test_name = argv[1];
  if (argv[2] == nullptr)
    optimization = "bayesian_optimization";
  else
    optimization = argv[2];

  std::string count = argv[3];
  std::string log_file = "hypermapper_taco_log.csv";

  bool log_exists = fs::exists(log_file);

  std::ofstream logger(log_file, std::ios_base::app);

  if(!log_exists) {
    logger << "Op,Size,Chunk size,Time" << std::endl;
  }

  // Set these values accordingly
  std::string OutputFoldername = "outdata";
  std::string AppName = "cpp_taco";
  int dimensionality_plus_one = 4;
  int NumSamples = dimensionality_plus_one;
  int NumIterations = NumSamples * 30;
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
  for (auto param : InParams) {
    std::cout << "Param: " << *param << "\n";
  }

  // Create json scenario
  std::string JSonFileNameStr =
      createjson(AppName, OutputFoldername, NumIterations,
                 NumSamples, InParams, Objectives, test_name, optimization, count);
  // exit(1);

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
      HMObjective Obj = calculateObjective(InParams, test_name, logger);  // Function to run hypermapper on
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

  std::cout << "No sched: " << no_sched_time << std::endl;

  FILE *fp;
  std::string cmdPareto("python3 ");
  cmdPareto += getenv("HYPERMAPPER_HOME");
  cmdPareto += "/scripts/plot_optimization_results.py -j";
  cmdPareto += " " + JSonFileNameStr;
  cmdPareto += " -i outdata -o " + test_name + "_plot.png";
  cmdPareto += " --expert_configuration " + to_string(default_config_time);
  cmdPareto += " -t " + op + " " + to_string(num_j) + " d:" + to_string(dimensionality_plus_one - 1) + " sparsity:" + to_string(sparsity);
  cmdPareto += " -doe ";
  // cmdPareto += " " + to_string(no_sched_time);
  std::cout << "Executing " << cmdPareto << std::endl;
  fp = popen(cmdPareto.c_str(), "r");
  while (fgets(buffer, max_buffer, fp))
    printf("%s", buffer);
  pclose(fp);

  logger.close();

  return 0;
}
