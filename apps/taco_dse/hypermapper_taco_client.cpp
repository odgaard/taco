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
#include <unordered_map>

#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/transformations.h"
#include "taco/lower/lower.h"

#include "hypermapper_taco_client.h"
#include "taco_helper.h"
#include "json.hpp"
#include "argparse/argparse.hpp"

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
int num_reps;

struct popen2 {
  pid_t child_pid;
  int from_child, to_child;
};

int popen2(const char *cmdline, struct popen2 *childinfo);
std::string createjson(std::string AppName, std::string OutputFoldername, int NumIterations,
                  int NumDSERandomSamples, std::vector<HMInputParamBase *> &InParams,
                  std::vector<std::string> Objectives, std::string optimization, std::string count);
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
HMObjective calculateObjectiveSpMVDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger);
HMObjective calculateObjectiveSpMVSparse(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger);
HMObjective calculateObjectiveSpMMDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger);
HMObjective calculateObjectiveSDDMMDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger);
HMObjective calculateObjectiveTTVDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger);
HMObjective calculateObjectiveTTMDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger);
HMObjective calculateObjective(std::vector<HMInputParamBase *> &InParams, std::string test_name, std::string matrix_name, std::ofstream &logger);
void spMMExhaustiveSearch();

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
                  std::vector<std::string> Objectives, std::string optimization, std::string count="0") {

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

  // read and load the json template
  json HMScenario;
  std::ifstream json_template(AppName + ".json", std::ifstream::binary);
  if (!json_template) {
    cout << "Failed to open json file " + AppName + ".json" << endl;
    exit(1);
  }
  json_template >> HMScenario;

  // set the dynamic features
  HMScenario["optimization_objectives"] = json(Objectives);
  HMScenario["run_directory"] = CurrentDir;
  HMScenario["log_file"] = OutputFoldername + "/log_" + AppName + ".log";
  if(optimization != "random_sampling") {
    HMScenario["optimization_method"] = optimization;
    HMScenario["optimization_iterations"] = NumIterations;
  }
  else {
    HMScenario["optimization_iterations"] = 0;
  }

  HMScenario["output_data_file"] =
      OutputFoldername + "/" + AppName + "_" +  optimization + count + "_output_data.csv";
  HMScenario["output_pareto_file"] =
      OutputFoldername + "/" + AppName + "_output_pareto.csv";
  HMScenario["output_image"]["output_image_pdf_file"] =
      OutputFoldername + "_" + AppName + "_output_image.pdf";

  // save the completed json file in the output directory
  ofstream HyperMapperScenarioFile;
  std::string JSonFileNameStr =
      CurrentDir + "/" + OutputFoldername + "/" + AppName + "_" + optimization + "_scenario.json";

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
  std::vector<int> chunkSizeValues{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
  std::vector<int> unrollFactorValues{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
  std::vector<int> ompChunkSizeValues{0, 1, 2, 4, 8, 16, 32, 64, 128};
  std::vector<int> ompNumThreadsValues{1, 2, 4, 8, 16, 32, 64};
  std::vector<int> ompSchedulingType{0, 1};

  HMInputParam<int> *chunkSizeParam = new HMInputParam<int>("chunk_size", ParamType::Ordinal);
  chunkSizeParam->setRange(chunkSizeValues);
  InParams.push_back(chunkSizeParam);
  numParams++;

  HMInputParam<int> *unrollFactorParam = new HMInputParam<int>("unroll_factor", ParamType::Ordinal);
  unrollFactorParam->setRange(unrollFactorValues);
  InParams.push_back(unrollFactorParam);
  numParams++;

  // FIXME: For some reason categorical fails for this param
  HMInputParam<int> *ompSchedulingTypeParam = new HMInputParam<int>("omp_scheduling_type", ParamType::Ordinal);
  ompSchedulingTypeParam->setRange(ompSchedulingType);
  InParams.push_back(ompSchedulingTypeParam);
  numParams++;

  HMInputParam<int> *ompChunkSizeParam = new HMInputParam<int>("omp_chunk_size", ParamType::Ordinal);
  ompChunkSizeParam->setRange(ompChunkSizeValues);
  InParams.push_back(ompChunkSizeParam);
  numParams++;

  HMInputParam<int> *ompNumThreadsParam = new HMInputParam<int>("omp_num_threads", ParamType::Ordinal);
  ompNumThreadsParam->setRange(ompNumThreadsValues);
  InParams.push_back(ompNumThreadsParam);
  numParams++;

  int reorder_size = 5;
  std::vector<std::vector<int>> valuesRange {std::vector<int>{reorder_size}};
  HMInputParam<std::vector<int>> *loopOrderingParam = new HMInputParam<std::vector<int>>("permutation", ParamType::Permutation);
  loopOrderingParam->setRange(valuesRange);
  InParams.push_back(loopOrderingParam);
  numParams++;
  num_loops = reorder_size;

  return numParams;
}

// Function that populates input parameters
int collectInputParamsSDDMM(std::vector<HMInputParamBase *> &InParams) {
  int numParams = 0;

  std::vector<int> chunkSizeValues{2, 4, 8, 16, 32, 64, 128, 256, 512};
  std::vector<int> unrollFactorValues{2, 4, 8, 16, 32, 64, 128, 256};
  std::vector<int> ompChunkSizeValues{0, 1, 2, 4, 8, 16, 32, 64, 128};
  std::vector<int> ompNumThreadsValues{1, 2, 4, 8, 16, 32, 64};
  std::vector<int> ompSchedulingType{0, 1};

  HMInputParam<int> *chunkSizeParam = new HMInputParam<int>("chunk_size", ParamType::Ordinal);
  chunkSizeParam->setRange(chunkSizeValues);
  InParams.push_back(chunkSizeParam);
  numParams++;

  HMInputParam<int> *unrollFactorParam = new HMInputParam<int>("unroll_factor", ParamType::Ordinal);
  unrollFactorParam->setRange(unrollFactorValues);
  InParams.push_back(unrollFactorParam);
  numParams++;

  // FIXME: For some reason categorical fails for this param
  HMInputParam<int> *ompSchedulingTypeParam = new HMInputParam<int>("omp_scheduling_type", ParamType::Ordinal);
  ompSchedulingTypeParam->setRange(ompSchedulingType);
  InParams.push_back(ompSchedulingTypeParam);
  numParams++;

  HMInputParam<int> *ompChunkSizeParam = new HMInputParam<int>("omp_chunk_size", ParamType::Ordinal);
  ompChunkSizeParam->setRange(ompChunkSizeValues);
  InParams.push_back(ompChunkSizeParam);
  numParams++;

  HMInputParam<int> *ompNumThreadsParam = new HMInputParam<int>("omp_num_threads", ParamType::Ordinal);
  ompNumThreadsParam->setRange(ompNumThreadsValues);
  InParams.push_back(ompNumThreadsParam);
  numParams++;

  int reorder_size = 5;
  std::vector<std::vector<int>> valuesRange {std::vector<int>{reorder_size}};
  HMInputParam<std::vector<int>> *loopOrderingParam = new HMInputParam<std::vector<int>>("permutation", ParamType::Permutation);
  loopOrderingParam->setRange(valuesRange);
  InParams.push_back(loopOrderingParam);
  numParams++;
  num_loops = reorder_size;

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
      case IntVector:
        delete static_cast<HMInputParam<std::vector<int>>*>(p);
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
      static_cast<HMInputParam<int>*>(Param)->setVal(std::stoi(ParamVal));
      break;
    case Float:
      static_cast<HMInputParam<float>*>(Param)->setVal(stof(ParamVal));
      break;
    case IntVector:
      // param val comes as a string such as: (2,1,4,3,0)
      // remove parenthesis, then transform to vector<int>
      ParamVal.pop_back();
      ParamVal.erase(ParamVal.begin());
      vector<int> convertedVector;
      string token;
      istringstream tokenStream(ParamVal);
      while (getline(tokenStream, token, ',')) {
        convertedVector.push_back(std::stoi(token));
      }
      static_cast<HMInputParam<std::vector<int>>*>(Param)->setVal(convertedVector);
      break;
  }
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveSpMVDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger) {
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

  // int NUM_I = 10000;
  // int NUM_J = 10000;
  if(!initialized) {
    // spmv_handler = new SpMV(NUM_I, NUM_J);
    spmv_handler = new SpMV();
    spmv_handler->matrix_name = matrix_name;
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
HMObjective calculateObjectiveSpMVSparse(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger) {
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
    spmv_sparse_handler->matrix_name = matrix_name;
    spmv_sparse_handler->initialize_data(1);
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

double median(vector<double> vec) {
        typedef vector<int>::size_type vec_sz;

        vec_sz size = vec.size();
        if (size == 0)
                throw domain_error("median of an empty vector");

        sort(vec.begin(), vec.end());

        vec_sz mid = size/2;

        return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveSpMMDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger) {

  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();
  int unroll_factor = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();
  int omp_scheduling_type = static_cast<HMInputParam<int>*>(InputParams[2])->getVal();
  int omp_chunk_size = static_cast<HMInputParam<int>*>(InputParams[3])->getVal();
  int omp_num_threads = static_cast<HMInputParam<int>*>(InputParams[4])->getVal();
  std::vector<int> loop_ordering = static_cast<HMInputParam<std::vector<int>>*>(InputParams[5])->getVal();

  std::vector<int> default_ordering{0,1,2,3,4};
  // int NUM_I = 67173;
  // int NUM_J = 67173;
  int NUM_K = 256;
  // float _sparsity = .982356;
  std::vector<double> compute_times;

  if(!initialized) {
    cout << "INITIALIZING" << endl;
    spmm_handler = new SpMM();
    spmm_handler->matrix_name = matrix_name;
    spmm_handler->NUM_K = NUM_K;
    spmm_handler->initialize_data(1);
    // result = spmm_handler->get_A();
    taco::Tensor<double> temp_result({spmm_handler->NUM_I, spmm_handler->NUM_K}, taco::dense);
    initialized = true;
    sparsity = spmm_handler->get_sparsity();
    num_j = spmm_handler->get_num_j();
    op = "SpMM";

    // Taco requires you to start with running the deafult
    // std::vector<int> tmp_loop_ordering = default_ordering;
    // int tmp_chunk_size = 16;
    // int tmp_unroll_factor = 8;
    // int tmp_omp_scheduling_type = 0;
    // int tmp_omp_chunk_size = 1;
    // int tmp_num_threads = 32;

    // spmm_handler->generate_schedule(temp_result, tmp_chunk_size, tmp_unroll_factor, tmp_loop_ordering, tmp_omp_scheduling_type, tmp_omp_chunk_size, tmp_num_threads);
    // compute_times = vector<double>();
    // for(int i = 0; i < num_reps; i++) {
    //   spmm_handler->compute(temp_result, true);
    //   compute_times.push_back(spmm_handler->get_compute_time());
    // }
    // default_config_time = median(compute_times);
  }


  compute_times = std::vector<double>();
  spmm_handler->set_cold_run();
  taco::Tensor<double> temp_result({spmm_handler->NUM_I, spmm_handler->NUM_K}, taco::dense);
  spmm_handler->generate_schedule(temp_result, chunk_size, unroll_factor, loop_ordering, omp_scheduling_type, omp_chunk_size, omp_num_threads);
  for(int i = 0; i < num_reps; i++) {
    // spmm_handler->schedule_and_compute(temp_result, chunk_size, unroll_factor, loop_ordering, omp_scheduling_type, omp_chunk_size, omp_num_threads, false);
    spmm_handler->compute(temp_result, false);
    compute_times.push_back(spmm_handler->get_compute_time());
    // std::cout << spmm_handler->get_compute_time() << std::endl;
  }

  Obj.compute_time = median(compute_times);
  if(default_config_time == 0.0f) {
    std::cout << "Default time: " << Obj.compute_time << std::endl;
    default_config_time = Obj.compute_time;
  }
  return Obj;
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveSDDMMDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();
  int unroll_factor = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();
  int omp_chunk_size = static_cast<HMInputParam<int>*>(InputParams[2])->getVal();
  int omp_scheduling_type = static_cast<HMInputParam<int>*>(InputParams[3])->getVal();
  std::vector<int> loop_ordering = static_cast<HMInputParam<std::vector<int>>*>(InputParams[4])->getVal();
  int num_threads = 32;
  std::vector<int> default_ordering{0,1,2,3,4};

  //Initialize tensors
  // int NUM_I = 10000;
  // int NUM_J = 1000;
  // int NUM_K = 10000;
  std::vector<double> compute_times;

  if(!initialized) {
    cout << "INITIALIZING" << endl;
    // sddmm_handler = new SDDMM(NUM_I, NUM_J, NUM_K);
    sddmm_handler = new SDDMM();
    sddmm_handler->matrix_name = matrix_name;
    sddmm_handler->initialize_data(1);
    initialized = true;
    sparsity = sddmm_handler->get_sparsity();
    num_j = sddmm_handler->get_num_j();
    op = "SDDMM";

    // Taco requires you to start with running the deafult
    std::vector<int> tmp_loop_ordering = default_ordering;
    int tmp_chunk_size = 16;
    int tmp_unroll_factor = 8;
    sddmm_handler->generate_schedule(tmp_chunk_size, tmp_unroll_factor, tmp_loop_ordering);
    compute_times = vector<double>();
    for(int i = 0; i < num_reps; i++) {
      sddmm_handler->compute(true);
      compute_times.push_back(sddmm_handler->get_compute_time());
    }
    default_config_time = median(compute_times);
  }

  //Initiate scheduling passing in chunk_size (param to optimize)
  sddmm_handler->generate_schedule(chunk_size, unroll_factor, loop_ordering, omp_scheduling_type, omp_chunk_size, num_threads);

  // bool default_config = (chunk_size == 16 && unroll_factor == 8 && order == 0);
  compute_times = vector<double>();
  for(int i = 0; i < num_reps; i++) {
    sddmm_handler->compute(false);
    compute_times.push_back(sddmm_handler->get_compute_time());
  }

  Obj.compute_time = median(compute_times);
  return Obj;
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveTTVDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger) {
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
HMObjective calculateObjectiveTTMDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger) {
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

HMObjective calculateObjective(std::vector<HMInputParamBase *> &InParams, std::string test_name, std::string matrix_name, std::ofstream &logger) {
  if (test_name == "SpMV")
    return calculateObjectiveSpMVDense(InParams, matrix_name, logger);
  if (test_name == "SpMVSparse")
    return calculateObjectiveSpMVSparse(InParams, matrix_name, logger);
  if (test_name == "SpMM")
    return calculateObjectiveSpMMDense(InParams, matrix_name, logger);
  if (test_name == "SDDMM")
    return calculateObjectiveSDDMMDense(InParams, matrix_name, logger);
  if (test_name == "TTV")
    return calculateObjectiveTTVDense(InParams, matrix_name, logger);
  if (test_name == "TTM")
    return calculateObjectiveTTMDense(InParams, matrix_name, logger);
  else {
    std::cout << "Test case not implemented yet" << std::endl;
    exit(-1);
  }
}

bool validate_ordering_sddmm(std::vector<int> order) {
  std::unordered_map<int, int> dict;
  for(int i = 0; i < order.size(); i++) {
    dict[order[i]] = i;
    if(order[i] == 0 || order[i] == 1) {
      if(dict.find(3) != dict.end()) {
        return false;
      }
    }
    if(order[i] == 3) {
      if(dict.size() == 1) {
        return false;
      }
      if(dict.find(2) != dict.end() && dict.find(4) != dict.end()) {
        return false;
      }
    }
    if(order[i] == 0) {
      if(dict.find(2) != dict.end()) {
        return false;
      }
    } 
    if(order[i] == 1) {
      if(dict.find(2) != dict.end()) {
        return false;
      }
    } 
  }
  return true;
}

void sddmmExhaustiveSearch(std::string matrix_name, std::ofstream &logger) {

  std::vector<std::vector<std::vector<double>>> obj_values(120, vector(7, vector<double>(2)));
  using namespace taco;

  std::vector<int> default_ordering{0,1,2,3,4};
  std::vector<double> compute_times;

  if(!initialized) {
    // spmm_handler = new SpMM(NUM_I, NUM_J, NUM_K, sparsity);
    // spmm_handler = new SpMM(0, NUM_I, NUM_J, NUM_K, _sparsity);
    sddmm_handler = new SDDMM();
    sddmm_handler->matrix_name = matrix_name;
    // spmm_handler->initialize_data(0);
    sddmm_handler->initialize_data(1);
    initialized = true;
    sparsity = sddmm_handler->get_sparsity();
    op = "SDDMM";

    // Taco requires you to start with running the deafult
    std::vector<int> tmp_loop_ordering = default_ordering;
    // int tmp_chunk_size = 16;
    // int tmp_unroll_factor = 8;
    // taco::Tensor<double> temp_result({sddmm_handler->NUM_I, sddmm_handler->NUM_J}, taco::dense);
    // sddmm_handler->generate_schedule(temp_result, tmp_chunk_size, tmp_unroll_factor, tmp_loop_ordering);
    // compute_times = vector<double>();
    // for(int i = 0; i < num_reps; i++) {
    //   sddmm_handler->compute(temp_result, true);
    //   compute_times.push_back(sddmm_handler->get_compute_time());
    // }
    // default_config_time = median(compute_times);
  }

  std::string test_name = "SDDMM";
  std::vector<int> chunkSizeValues{8, 16, 32, 64, 128, 256, 512};
  std::vector<int> unrollFactorValues{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
  std::vector<int> ompSchedulingType{0, 1};
  std::vector<int> ompChunkSizeValues{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
  std::vector<int> loop_ordering{0, 1, 2, 3, 4};
  int unroll_factor = 8;
  int chunk_size = 16;
  int num_threads = 32;


  int permutation_idx = 0;
  std::vector<bool> valid(120, true);
  for (int omp_chunk_size_idx = 0; omp_chunk_size_idx < 7; omp_chunk_size_idx++) {
    int omp_chunk_size = ompChunkSizeValues[omp_chunk_size_idx];
    cout << chunk_size << endl;

    permutation_idx = 0;
    loop_ordering = vector<int>{0, 1, 2, 3, 4};

    do {
      for (int l : loop_ordering) {
        cout << l << " ";
      }
      cout << endl;
      for (int omp_scheduling_type = 0; omp_scheduling_type < 2; omp_scheduling_type++) {

        sddmm_handler->set_cold_run();
        double total_time = 0.0;
        for(int i = 0; i < num_reps; i++) {
          taco::Tensor<double> temp_result({sddmm_handler->NUM_I, sddmm_handler->NUM_J}, taco::dense);
          try {
            sddmm_handler->schedule_and_compute(temp_result, chunk_size, unroll_factor, loop_ordering, omp_scheduling_type, omp_chunk_size, num_threads, false);
            // sddmm_handler->compute(temp_result, false);
            total_time += sddmm_handler->get_compute_time();
          } catch(const taco::TacoException& err) {
            std::cout << "Exception found" << std::endl;
            total_time = 0.0;
            valid[permutation_idx] = false;
            break;
          }
          // taco::TensorStorage store = temp_result.getStorage();
          // delete store;

        }
        // if(permutation_idx == 1) {
        //   goto end;
        // }

        double compute_time = total_time / num_reps;
        obj_values[permutation_idx][omp_chunk_size_idx][omp_scheduling_type] = compute_time;
      }
      permutation_idx ++;
    } while (std::next_permutation(loop_ordering.begin(), loop_ordering.end()));
  }
  end:

  std::vector<int> loop_ordering2{0, 1, 2, 3, 4};
  permutation_idx = 0;

  do {
    for (int omp_scheduling_type = 0; omp_scheduling_type < 2; omp_scheduling_type++) {
      for (int chunkSize_idx = 0; chunkSize_idx < 7; chunkSize_idx++) {
        cout << obj_values[permutation_idx][chunkSize_idx][omp_scheduling_type] << " ";
      }
    }
    cout << endl;
    permutation_idx ++;
  } while (std::next_permutation(loop_ordering2.begin(), loop_ordering2.end()));

  std::vector<int> loop_ordering3{0, 1, 2, 3, 4};
  permutation_idx = 0;

  do {
    // if(!validate_ordering(loop_ordering3)) {
    //     continue;
    // }
    for(auto elems : loop_ordering3) {
      std::cout << elems << " ";
    }
    cout << "\t" << valid[permutation_idx];
    cout << endl;
    permutation_idx ++;
  } while (std::next_permutation(loop_ordering3.begin(), loop_ordering3.end()));
}

bool validate_ordering(std::vector<int> order) {
  std::unordered_map<int, int> dict;
  for(int i = 0; i < order.size(); i++) {
    dict[order[i]] = i;
    if(order[i] == 0 || order[i] == 1) {
      if(dict.find(3) != dict.end()) {
        return false;
      }
    }
    if(order[i] == 3) {
      if(dict.size() == 1) {
        return false;
      }
      if(dict.find(2) != dict.end() && dict.find(4) != dict.end()) {
        return false;
      }
    }
    if(order[i] == 0) {
      if(dict.find(2) != dict.end()) {
        return false;
      }
    } 
    if(order[i] == 1) {
      if(dict.find(2) != dict.end()) {
        return false;
      }
    } 
  }
  return true;
}

void spmmExhaustiveSearch(std::string matrix_name, std::ofstream &logger) {

  std::vector<std::vector<std::vector<double>>> obj_values(120, vector(7, vector<double>(2)));
  using namespace taco;

  // int NUM_I = 67173;
  // int NUM_J = 67173;
  // int NUM_K = 256;
  // float _sparsity = .982356;
  std::vector<int> default_ordering{0,1,2,3,4};
  std::vector<double> compute_times;

  if(!initialized) {
    // spmm_handler = new SpMM(NUM_I, NUM_J, NUM_K, sparsity);
    // spmm_handler = new SpMM(0, NUM_I, NUM_J, NUM_K, _sparsity);
    spmm_handler = new SpMM();
    spmm_handler->matrix_name = matrix_name;
    // spmm_handler->initialize_data(0);
    spmm_handler->initialize_data(1);
    initialized = true;
    sparsity = spmm_handler->get_sparsity();
    op = "SpMM";

    // Taco requires you to start with running the deafult
    std::vector<int> tmp_loop_ordering = default_ordering;
    int tmp_chunk_size = 16;
    int tmp_unroll_factor = 8;
    taco::Tensor<double> temp_result({spmm_handler->NUM_I, spmm_handler->NUM_K}, taco::dense);
    spmm_handler->generate_schedule(temp_result, tmp_chunk_size, tmp_unroll_factor, tmp_loop_ordering, 0, 0, 1);
    compute_times = vector<double>();
    for(int i = 0; i < num_reps; i++) {
      spmm_handler->compute(temp_result, true);
      compute_times.push_back(spmm_handler->get_compute_time());
    }
    default_config_time = median(compute_times);
  }

  std::string test_name = "SpMM";
  std::vector<int> chunkSizeValues{2, 4, 8, 16, 32, 64, 128, 256, 512};
  std::vector<int> unrollFactorValues{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
  std::vector<int> ompSchedulingType{0, 1};
  std::vector<int> ompChunkSizeValues{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
  std::vector<int> loop_ordering{0, 1, 2, 3, 4};
  int unroll_factor = 8;
  int chunk_size = 16;
  int num_threads = 32;

  // std::cout << "before actual" << std::endl;

  std::vector<bool> valid(120, true);

  int permutation_idx = 0;
  for (int omp_chunk_size_idx = 0; omp_chunk_size_idx < 7; omp_chunk_size_idx++) {
    int omp_chunk_size = ompChunkSizeValues[omp_chunk_size_idx];
    cout << chunk_size << endl;

    permutation_idx = 0;
    loop_ordering = vector<int>{0, 1, 2, 3, 4};

    int count = 0;

    do {
      for (int l : loop_ordering) {
        cout << l << " ";
      }
      cout << endl;
      // if(!validate_ordering(loop_ordering)) {
      //   std::cout << "Invalid loop ordering" << std::endl;
      //   continue;
      // }
      for (int omp_scheduling_type = 0; omp_scheduling_type < 2; omp_scheduling_type++) {
        taco::Tensor<double> temp_result({spmm_handler->NUM_I, spmm_handler->NUM_K}, taco::dense);
        spmm_handler->generate_schedule(temp_result, chunk_size, unroll_factor, loop_ordering, omp_scheduling_type, omp_chunk_size, num_threads);
        spmm_handler->set_cold_run();

        double total_time = 0.0;
        for(int i = 0; i < num_reps; i++) {
          try {
            spmm_handler->compute(temp_result, false);
            total_time += spmm_handler->get_compute_time();
          // } catch (const std::runtime_error& err) {
          } catch (const taco::TacoException& err) {
            std::cout << "Exception found" << std::endl;
            // std::cout << err << std::endl;
            total_time = 0.0;
            valid[permutation_idx] = false;
            break;
          }
        }
        // exit(1);

        double compute_time = total_time / num_reps;
        obj_values[permutation_idx][omp_chunk_size_idx][omp_scheduling_type] = compute_time;
      }
      permutation_idx ++;
    } while (std::next_permutation(loop_ordering.begin(), loop_ordering.end()));
  }

  std::vector<int> loop_ordering2{0, 1, 2, 3, 4};
  permutation_idx = 0;

  do {
    if(!validate_ordering(loop_ordering2)) {
        continue;
    }
    for (int omp_scheduling_type = 0; omp_scheduling_type < 2; omp_scheduling_type++) {
      for (int chunkSize_idx = 0; chunkSize_idx < 7; chunkSize_idx++) {
        cout << obj_values[permutation_idx][chunkSize_idx][omp_scheduling_type] << " ";
      }
    }
    cout << endl;
    permutation_idx ++;
  } while (std::next_permutation(loop_ordering2.begin(), loop_ordering2.end()));

  std::vector<int> loop_ordering3{0, 1, 2, 3, 4};
  permutation_idx = 0;

  do {
    if(!validate_ordering(loop_ordering3)) {
        continue;
    }
    for(auto elems : loop_ordering3) {
      std::cout << elems << " ";
    }
    cout << "\t" << valid[permutation_idx];
    cout << endl;
    permutation_idx ++;
  } while (std::next_permutation(loop_ordering3.begin(), loop_ordering3.end()));
}

// void SpMMVarianceTest(std::ofstream &logger) {
//    using namespace taco;

//   int NUM_I = 67173;
//   int NUM_J = 67173;
//   int NUM_K = 1000;
//   float _sparsity = .982356;

//   std::vector<std::vector<double>> obj_vals;
//   std::vector<double> chunk_sizes{16, 8, 4, 32, 64, 512, 1024};
//   std::vector<double> unroll_factors{8, 4, 1, 2, 32, 4, 512};
//   std::vector<std::vector<int>> permutations{{0,1,2,3,4},
//                                              {1,4,2,3,0},
//                                              {4,3,2,1,0},
//                                              {2,3,1,4,0},
//                                              {4,2,3,1,0},
//                                              {3,0,2,4,1},
//                                              {0,1,4,2,3}};


//   if(!initialized) {
//     // spmm_handler = new SpMM(0, NUM_I, NUM_J, NUM_K, _sparsity);
//     spmm_handler = new SpMM();
//     // spmm_handler->initialize_data(0);
//     spmm_handler->initialize_data(1);
//     initialized = true;
//     sparsity = spmm_handler->get_sparsity();
//     num_j = spmm_handler->get_num_j();
//     op = "SpMM";

//     // Taco requires you to start with running the deafult
//     obj_vals.push_back(vector<double>());
//     spmm_handler->generate_schedule(chunk_sizes[0], unroll_factors[0], permutations[0]);

//     for(int i = 0; i < num_reps; i++) {
//       spmm_handler->compute(true);
//       obj_vals[0].push_back(spmm_handler->get_compute_time());
//     }
//   }

//   for (int i = 1; i < 7; i++) {
//     obj_vals.push_back(vector<double>());
//     spmm_handler->generate_schedule(chunk_sizes[i], unroll_factors[i], permutations[i]);
//     for(int j = 0; j < num_reps; j++) {
//       spmm_handler->compute(false);
//       obj_vals[i].push_back(spmm_handler->get_compute_time());
//     }
//   }

//   for (int i = 0; i < 7; i++) {
//     cout << chunk_sizes[i] << " " << unroll_factors[i] << " ";
//     for (int j = 0; j < 5; j++) {
//       cout << permutations[i][j] << " ";
//     }
//     cout << endl;
//   }
//   cout << endl;
//   for (int i = 0; i < 7; i++) {
//     for(int j = 0; j < num_reps; j++) {
//       cout << obj_vals[i][j] << " ";
//     }
//     cout << endl;
//   }
// }


int main(int argc, char **argv) {

  if (!getenv("HYPERMAPPER_HOME")) {
    std::string ErrMsg = "Environment variables are not set!\n";
    ErrMsg += "Please set HYPERMAPPER_HOME before running this ";
    setenv("HYPERMAPPER_HOME", "/home/ubuntu/hypermapper_dev", true);
    printf("Setting HM variable\n");
    // fatalError(ErrMsg);
  }

  taco::taco_set_num_threads(32);

  argparse::ArgumentParser program("./bin/taco-taco_dse");

  program.add_argument("-e", "--exhaustive")
    .help("Run exhaustive search")
    .default_value(false)
    .implicit_value(true);
  program.add_argument("-o", "--op")
    .help("TACO operation to run")
    .default_value(std::string("SpMM"))
    .required();
  program.add_argument("-m", "--method")
    .help("Hypermapper optimization method")
    .default_value(std::string("bayesian_optimization"))
    .required();
  program.add_argument("-n", "--num_reps")
    .help("Number of compute repetitions")
    .default_value(5)
    .scan<'i', int>()
    .required();
  program.add_argument("-mat", "--matrix_name")
    .help("Matrix to be used: auto or matrix market file")
    .default_value(std::string("auto"))
    .required();
  program.add_argument("-c", "--count")
    .help("Hypermapper run index; used to keep track of test result rep")
    .default_value(std::string("0"))
    .required();
  try {
    program.parse_args(argc, argv);    
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  // std::string test_name = "SpMM";
  std::string test_name, optimization, matrix_name, count;
  bool exh_search;
  test_name = program.get<std::string>("--op");
  optimization = program.get<std::string>("--method");
  matrix_name = program.get<std::string>("--matrix_name");
  num_reps = program.get<int>("--num_reps");
  exh_search = program.get<bool>("--exhaustive");
  count = program.get<std::string>("--count");

  std::string log_file = "hypermapper_taco_log.csv";
  bool log_exists = fs::exists(log_file);

  std::ofstream logger(log_file, std::ios_base::app);

  if(!log_exists) {
    logger << "Op,Size,Chunk size,Time" << std::endl;
  }

  if (exh_search) {
    // spmmExhaustiveSearch(matrix_name, logger);
    sddmmExhaustiveSearch(matrix_name, logger);
    // SpMMVarianceTest(logger);
    exit(1);
  }

  // Set these values accordingly

  std::string OutputFoldername;
  if (matrix_name == "auto") {
    OutputFoldername = "outdata_" + test_name;
  } else {
    // remove the file-ending of the matrix name add it to the output folder name
    size_t lastindex = matrix_name.find_last_of(".");
    string rawname = matrix_name.substr(0, lastindex);
    OutputFoldername = "outdata_" + test_name + "_" + rawname;
  }
  std::string AppName = "cpp_taco_" + test_name;
  int dimensionality_plus_one = 4;
  int NumSamples = dimensionality_plus_one;
  int NumIterations = 70;
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

  const int max_buffer = 1000;
  char buffer[max_buffer];
  std::string JSonFileNameStr;

  // Create json scenario
  JSonFileNameStr =
      createjson(AppName, OutputFoldername, NumIterations,
                 NumSamples, InParams, Objectives, optimization, count);

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
  cout << "opened hypermapper" << endl;

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
    std::cout << "numreq: " << NumReqStr << std::endl;
    int numRequests = std::stoi(NumReqStr);
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
        if (bufferStr.at(pos) == '(') {
          len = bufferStr.find_first_of(")\n", pos) - pos + 1;
        }
        std::string ParamValStr = bufferStr.substr(pos, len);
        setInputValue(InputParamsMap[param], ParamValStr);
        response += ParamValStr;
        response += ",";
        pos = bufferStr.find_first_of(",\n", pos + len) + 1;
      }
      HMObjective Obj = calculateObjective(InParams, test_name, matrix_name, logger);  // Function to run hypermapper on
      response += std::to_string(Obj.compute_time);
      response += "\n";
    }
    std::cout << "Response:\n" << response;
    fputs(response.c_str(), outstream);
    fflush(outstream);
    i++;
  }
  cout << "closing pipes" << endl;
  close(hypermapper.from_child);
  close(hypermapper.to_child);
  deleteInputParams(InParams);
  std::cout << "No sched: " << no_sched_time << std::endl;

  cout << JSonFileNameStr << endl;

  FILE *fp;
  std::string cmdPareto("python3 ");
  cmdPareto += getenv("HYPERMAPPER_HOME");
  cmdPareto += "/scripts/plot_optimization_results.py -j";
  cmdPareto += " " + JSonFileNameStr;
  cmdPareto += " -i " + OutputFoldername + " -o " + test_name + "_plot.png";
  cmdPareto += " --expert_configuration " + to_string(default_config_time);
  cmdPareto += " -t '" + op + " " + to_string(num_j) + " d:" + to_string(dimensionality_plus_one - 1) + " sparsity:" + to_string(sparsity) + "'";
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
