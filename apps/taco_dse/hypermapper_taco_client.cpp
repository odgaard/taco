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

#include <iostream>
#include <memory>
#include <string>
#include <grpcpp/grpcpp.h>
#include <typeinfo>
#include <condition_variable>
#include <mutex>
#include "config_service.grpc.pb.h" // The generated header from the .proto file
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <thread>

SpMV *spmv_handler;
SpMV *spmv_sparse_handler;
SpMM *spmm_handler;
SDDMM *sddmm_handler;
TTV *ttv_handler;
TTM *ttm_handler;
MTTKRP *mttkrp_handler;
bool initialized = false;

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;

using namespace std::chrono;
using json = nlohmann::json;

namespace fs = std::experimental::filesystem;
int WARP_SIZE = 32;
float default_config_time = 0.0f;
taco::util::TimeResults no_sched_results;
double no_sched_time = 0.0f;
std::vector<double> no_sched_times;
double no_sched_energy = 0.0f;
std::vector<double> no_sched_energies;
int num_loops = 0;
float sparsity = 0.0f;
int num_i = 0;
int num_j = 0;
std::string op;
int num_reps;

std::condition_variable shutdown_cv;
std::mutex shutdown_mutex;
bool shutdown_flag = false;

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
std::vector<HMInputParamBase *>::iterator findHMParamByKey(std::vector<HMInputParamBase *> &InParams, const std::string& Key);
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
std::string paramValueToString(HMInputParamBase *Param);



template <typename T>
void setParameterValue(HMInputParamBase *param, const T &value) {
            using ValueType = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<ValueType, int32_t>) {
                static_cast<HMInputParam<int>*>(param)->setVal(value);
            } else if constexpr (std::is_same_v<ValueType, float>) {
                static_cast<HMInputParam<float>*>(param)->setVal(value);
            } else if constexpr (std::is_same_v<ValueType, std::string>) {
                static_cast<HMInputParam<std::string>*>(param)->setVal(value);
            } else if constexpr (std::is_same_v<ValueType, google::protobuf::RepeatedField<int>>) {
                std::vector<int> vecValue(value.begin(), value.end());
                HMInputParam<std::vector<int>>* typed_param = static_cast<HMInputParam<std::vector<int>>*>(param);
                typed_param->setVal(vecValue);
            } else {
                std::cout << "Unknown set parameter type: " << typeid(ValueType).name() << std::endl;
            }
            // ... Add more types as needed
    }

std::ostream& operator<<(std::ostream& os, const google::protobuf::RepeatedField<int>& values) {
    for (int i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != values.size() - 1) {
            os << ", ";
        }
    }
    return os;
}

template <typename T>
void printParameter(const std::string& paramName, const T &value, const std::string& type) {
    std::cout << paramName << ": ";
    if constexpr (std::is_same_v<std::decay_t<decltype(value)>, std::vector<int>>) {
        for (size_t i = 0; i < value.size(); ++i) {
            std::cout << value[i];
            if (i != value.size() - 1) {
                std::cout << ", ";
            }
        }
    } else {
        std::cout << value;
    }
    std::cout << " (" << type << ")" << std::endl;
}

std::string paramValueToString(HMInputParamBase *Param) {
    switch (Param->getDType()) {
        case Int:
            return std::to_string(static_cast<HMInputParam<int>*>(Param)->getVal());
        case Float:
            return std::to_string(static_cast<HMInputParam<float>*>(Param)->getVal());
        case IntVector: {
            std::vector<int> values = static_cast<HMInputParam<std::vector<int>>*>(Param)->getVal();
            std::stringstream ss;
            ss << "(";
            for (size_t i = 0; i < values.size(); ++i) {
                ss << values[i];
                if (i != values.size() - 1) {
                    ss << ",";
                }
            }
            ss << ")";
            return ss.str();
        }
        default:
            return "Unknown Type";
    }
}

void printHMInputParamBaseVector(const std::vector<HMInputParamBase*>& vec) {
    for (const auto& param : vec) {
        std::cout << paramValueToString(param) << " ";
        std::cout << param->getDType() << " ";
    }
    std::cout << std::endl;
}

using Matrix = std::vector<std::vector<double>>;

Matrix generateRandomMatrix(int rows, int cols) {
    Matrix mat(rows, std::vector<double>(cols, 0.0));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i][j] = rand() % 10;
        }
    }
    return mat;
}

void gemm_mock(const Matrix& A, Matrix& B, Matrix& C) {
    int m = A.size();
    int n = B[0].size();
    int k = A[0].size();

    if (B.size() != k || C.size() != m || C[0].size() != n) {
        std::cerr << "Does not match" << std::endl;
        return;
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += A[i][p] * B[p][j];
            }
            C[i][j] = sum;
        }
    }
}

HMObjective mock_function() {
    int m = 500;
    int n = 500;
    int k = 500;
    taco::util::Timer timer;
    for (int i = 0; i < 5; i++) {
        timer.start();
        Matrix A = generateRandomMatrix(m, k);
        Matrix B = generateRandomMatrix(k, n);
        Matrix C(m, std::vector<double>(n, 0.0));

        gemm_mock(A, B, C);
        timer.stop();
    }
    auto time_result = timer.getResult();
    HMObjective obj;
    obj.results = time_result;
    return obj;
}

class ConfigurationServiceImpl final : public ConfigurationService::Service {
private:
    std::vector<HMInputParamBase *>& m_InParams;
    std::string m_test_name;
    std::string m_matrix_name;
    std::ofstream& m_logger; // Assuming logger is an ofstream. Adjust the type if needed.
    
public:
    ConfigurationServiceImpl(std::vector<HMInputParamBase *>& InParams, std::string test_name, std::string matrix_name, std::ofstream &logger) 
        : m_InParams(InParams), m_test_name(test_name), m_matrix_name(matrix_name), m_logger(logger) {}

    Status RunConfigurationsClientServer(ServerContext* context, 
                                         const ConfigurationRequest* request, 
                                         ConfigurationResponse* response) override {
        char hostname[HOST_NAME_MAX];
        char username[LOGIN_NAME_MAX];
        int result_code = gethostname(hostname, HOST_NAME_MAX);
        // Access and process the configurations:
        const Configuration& config = request->configurations();
        //std::cout << "[" << hostname << "]: " << "Config received" << std::endl;
        // Inside your loop:
        for (const auto& param : config.parameters()) {
            const std::string& param_name = param.first;
            const Parameter& parameter = param.second;
            auto it = findHMParamByKey(m_InParams, param_name);
            if (it == m_InParams.end()) {
              return Status::CANCELLED;
            }
            HMInputParamBase* hmParam = *it;

            if (parameter.has_integer_param()) {
                setParameterValue(hmParam, parameter.integer_param().value());
                //printParameter(param_name, parameter.integer_param().value(), "Integer");
            } else if (parameter.has_real_param()) {
                setParameterValue(hmParam, parameter.real_param().value());
                //printParameter(param_name, parameter.real_param().value(), "Real");
            } else if (parameter.has_categorical_param()) {
                setParameterValue(hmParam, parameter.categorical_param().value());
                //printParameter(param_name, parameter.categorical_param().value(), "Categorical");
            } else if (parameter.has_ordinal_param()) {
                setParameterValue(hmParam, parameter.ordinal_param().value());
                //printParameter(param_name, parameter.ordinal_param().value(), "Ordinal");
            } else if (parameter.has_string_param()) {
                setParameterValue(hmParam, parameter.string_param().value());
                //printParameter(param_name, parameter.string_param().value(), "String");
            } else if (parameter.has_permutation_param()) {
                setParameterValue(hmParam, parameter.permutation_param().values());
                //printParameter(param_name, parameter.permutation_param().values(), "Permutation");
            } else {
              return Status::CANCELLED;
            }
        }

        //std::cout << "Received output_data_file: " << request->output_data_file() << std::endl;
        //printHMInputParamBaseVector(m_InParams);

        HMObjective obj;
        std::vector<double> temp_meds;
        double temp_med;
        //for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 5; i++) {
          try {
              //obj = mock_function();
              obj = calculateObjective(m_InParams, m_test_name, m_matrix_name, m_logger);
              temp_med = med(obj.results.times);
              temp_meds.push_back(temp_med);
          } catch (const std::exception& e) {
              m_logger << "Exception caught: " << e.what() << std::endl;
              return Status(StatusCode::INTERNAL, e.what());
          } catch (...) {
              m_logger << "Unknown exception caught" << std::endl;
              return Status(StatusCode::INTERNAL, "Unknown error");
          }
          std::cout << temp_med << std::endl;
          std::this_thread::sleep_for(std::chrono::seconds(15));
        }
        //std::this_thread::sleep_for(std::chrono::seconds(14));
        //}
        //std::this_thread::sleep_for(std::chrono::seconds(10));

        //auto min_iter = std::min_element(temp_meds.begin(), temp_meds.end());
        double new_med = med(temp_meds);
        //double new_med = *min_iter;
      
        // Create a mocked response:
        taco::util::TimeResults local_results = obj.results;
        std::cout << "[" << hostname << "]: " << local_results.median << ", " << med(local_results.energy_consumptions) << std::endl;
        Metric metric_median_time;
        metric_median_time.add_values(new_med);
        //metric_median_time.add_values(med(local_results.times));
        response->add_metrics()->CopyFrom(metric_median_time);
        Metric metric_compute_times;
        for (double time : local_results.times) {
            metric_compute_times.mutable_values()->Add(time);
        }
        response->add_metrics()->CopyFrom(metric_compute_times);


        //Metric metric_median_energy;
        //metric_median_energy.add_values(med(local_results.energy_consumptions));
        //response->add_metrics()->CopyFrom(metric_median_energy);
        Metric metric_energy_consumptions;
        for (double energy_consumption : local_results.energy_consumptions) {
            metric_energy_consumptions.mutable_values()->Add(energy_consumption);
        }
        response->add_metrics()->CopyFrom(metric_energy_consumptions);


        // Get current time in milliseconds since the epoch
        auto now = std::chrono::system_clock::now();
        auto epoch = now.time_since_epoch();
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);

        Timestamp timestamp;
        timestamp.set_timestamp(milliseconds.count()); 
        response->mutable_timestamps()->CopyFrom(timestamp);

        Feasible feasible;
        feasible.set_value(true); // Mocked feasibility value
        response->mutable_feasible()->CopyFrom(feasible);

        std::this_thread::sleep_for(std::chrono::seconds(10));

        return Status::OK;
    }
    grpc::Status Shutdown(grpc::ServerContext* context, const ShutdownRequest* request,
                          ShutdownResponse* response) override {
        if (request->shutdown()) {
            std::unique_lock<std::mutex> lock(shutdown_mutex);
            shutdown_flag = true;  // Set the shutdown flag
            shutdown_cv.notify_one();  // Notify the waiting server thread
        }
        response->set_success(true);  // Acknowledge the shutdown request
        return grpc::Status::OK;
    }
};



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
                  std::vector<std::string> Objectives, bool predictor, std::string optimization, std::string count="0") {

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
  //HMScenario["log_file"] = OutputFoldername + "/log_" + AppName + "_" + count + ".log";
  if(optimization != "random_sampling" && optimization != "embedding_random_sampling") {
    HMScenario["optimization_method"] = optimization;
    HMScenario["optimization_iterations"] = NumIterations;
  }
  else {
    HMScenario["optimization_iterations"] = 0;
  }

  if (predictor){
    json HMFeasibleOutput;
    HMFeasibleOutput["enable_feasible_predictor"] = true;
    HMFeasibleOutput["false_value"] = "0";
    HMFeasibleOutput["true_value"] = "1";
    HMScenario["feasible_output"] = HMFeasibleOutput;
  }

  json HMDOE;
  HMDOE["doe_type"] = "random sampling";
  HMDOE["number_of_samples"] = NumDSERandomSamples;
  if(optimization == "random_sampling" || optimization == "embedding_random_sampling") {
    HMDOE["number_of_samples"] = NumDSERandomSamples + NumIterations;
  }
  if(optimization == "embedding_random_sampling") {
      HMDOE["doe_type"] = "embedding random sampling";
  }
  HMScenario["design_of_experiment"] = HMDOE;

  HMScenario["output_data_file"] =
      OutputFoldername + "/" + AppName + "_" +  optimization + count + "_output_data.csv";

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

void addCommonParams(std::vector<HMInputParamBase *> &InParams) {
  InParams.push_back(new HMInputParam<int>("omp_scheduling_type", ParamType::Ordinal, {0, 1}));
  InParams.push_back(new HMInputParam<int>("omp_chunk_size", ParamType::Ordinal, {0, 1, 2, 4, 8, 16, 32, 64, 128}));
  InParams.push_back(new HMInputParam<int>("omp_num_threads", ParamType::Ordinal,[]
    {
      std::vector<int> values;
      for(int i = 1; i <= 64; ++i) {
          if(i % 2 == 0 || i == 1) {
              values.push_back(i);
          }
      }
      return values;
    }()  // The () at the end causes the lambda to be called immediately
  ));

    //InParams.push_back(new HMInputParam<int>("omp_num_threads", ParamType::Ordinal, {1, 2, 4, 8, 16, 32, 64}));

  int reorder_size = 5;
  InParams.push_back(new HMInputParam<std::vector<int>>("permutation", ParamType::Permutation, {std::vector<int>{reorder_size}}));
  num_loops = reorder_size;
}

int collectInputParamsSpMV(std::vector<HMInputParamBase *> &InParams, int SPLIT = 0) {
  InParams.push_back(new HMInputParam<int>("chunk_size", ParamType::Ordinal, {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}));
  InParams.push_back(new HMInputParam<int>("chunk_size2", ParamType::Ordinal, {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}));
  InParams.push_back(new HMInputParam<int>("chunk_size3", ParamType::Ordinal, {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}));
  addCommonParams(InParams);
  return InParams.size();
}
int collectInputParamsSpMM(std::vector<HMInputParamBase *> &InParams) {
  InParams.push_back(new HMInputParam<int>("chunk_size", ParamType::Ordinal, {1, 2, 4, 8, 16, 32, 64, 128, 256}));
  InParams.push_back(new HMInputParam<int>("unroll_factor", ParamType::Ordinal, {1, 2, 4, 8}));
  addCommonParams(InParams);
  return InParams.size();
}

int collectInputParamsSDDMM(std::vector<HMInputParamBase *> &InParams) {
  InParams.push_back(new HMInputParam<int>("chunk_size", ParamType::Ordinal, {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}));
  InParams.push_back(new HMInputParam<int>("unroll_factor", ParamType::Ordinal, {2, 4, 8, 16, 32, 64, 128, 256}));
  addCommonParams(InParams);
  return InParams.size();
}

int collectInputParamsTTV(std::vector<HMInputParamBase *> &InParams) {
  InParams.push_back(new HMInputParam<int>("chunk_size_i", ParamType::Ordinal, {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}));
  InParams.push_back(new HMInputParam<int>("chunk_size_fpos", ParamType::Ordinal, {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}));
  InParams.push_back(new HMInputParam<int>("chunk_size_k", ParamType::Ordinal, {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}));
  addCommonParams(InParams);
  return InParams.size();
}

int collectInputParamsTTM(std::vector<HMInputParamBase *> &InParams) {
  InParams.push_back(new HMInputParam<int>("chunk_size", ParamType::Ordinal, {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}));
  InParams.push_back(new HMInputParam<int>("unroll_factor", ParamType::Ordinal, {2, 4, 8, 16, 32, 64, 128, 256}));
  addCommonParams(InParams);
  return InParams.size();
}

int collectInputParamsMTTKRP(std::vector<HMInputParamBase *> &InParams) {
  InParams.push_back(new HMInputParam<int>("chunk_size", ParamType::Ordinal, {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}));
  InParams.push_back(new HMInputParam<int>("unroll_factor", ParamType::Ordinal, {2, 4, 8, 16, 32, 64, 128, 256}));
  addCommonParams(InParams);
  return InParams.size();
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
  if (test_name == "MTTKRP")
    return collectInputParamsMTTKRP(InParams);
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

std::vector<HMInputParamBase *>::iterator findHMParamByKey(std::vector<HMInputParamBase *> &InParams, const std::string& Key) {
    for (auto it = InParams.begin(); it != InParams.end(); ++it) {
        HMInputParamBase* Param = *it;
        if (*Param == Key) {
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
HMObjective calculateObjectiveSpMVDense_(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger) {
  using namespace taco;
  HMObjective Obj;
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
  int ITERATIONS = 1; // Was 10
  // float _sparsity = .982356;
  std::vector<double> compute_times;

  bool no_sched_init = false;

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
    num_i = spmm_handler->get_num_i();
    num_j = spmm_handler->get_num_j();
    op = "SpMM";

    taco::util::Timer local_timer;
    for(int i = 0; i < 5; i++) { // I used to be equal to 5
        local_timer = spmm_handler->compute_unscheduled(local_timer);
    }

    auto result = local_timer.getResult();
    no_sched_results = result;
    no_sched_time = result.median;
    no_sched_times = result.times;
    no_sched_energy = median(result.energy_consumptions);
    no_sched_energies = result.energy_consumptions;
    no_sched_init = true;
  }


  compute_times = std::vector<double>();
  spmm_handler->set_cold_run();
  taco::Tensor<double> temp_result({spmm_handler->NUM_I, spmm_handler->NUM_K}, taco::dense);


  if(default_config_time == 0.0f) {
    // std::cout << "Default time: " << Obj.compute_time << std::endl;
    int temp_chunk_size = 16;
    int temp_unroll_factor = 8;
    std::vector<int> temp_loop_ordering{0,1,2,3,4};
    int temp_omp_scheduling_type = 0;
    int temp_omp_chunk_size = 1;
    int temp_omp_num_threads = 32;

    spmm_handler->schedule_and_compute(temp_result, temp_chunk_size, temp_unroll_factor, temp_loop_ordering, temp_omp_scheduling_type, temp_omp_chunk_size, temp_omp_num_threads, false, ITERATIONS);
    spmm_handler->set_cold_run();

    default_config_time = spmm_handler->get_compute_time();
    std::cout << spmm_handler->get_num_i() << "," << spmm_handler->get_num_j() << "," << default_config_time << "," << no_sched_time << std::endl;
    logger << spmm_handler->get_num_i() << "," << spmm_handler->get_num_j() << "," << default_config_time << "," << no_sched_time << std::endl;
  }

  if(!no_sched_init) {
    spmm_handler->schedule_and_compute(temp_result, chunk_size, unroll_factor, loop_ordering, omp_scheduling_type, omp_chunk_size, omp_num_threads, false, ITERATIONS);
    spmm_handler->set_cold_run();
  }

  if (no_sched_init) {
      Obj.results = no_sched_results;
  } else {
      taco::util::TimeResults local_results = spmm_handler->get_results();
      Obj.results = local_results;
  }


  return Obj;
}

HMObjective calculateObjectiveSpMVDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger) {

  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();
  int chunk_size2 = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();
  int chunk_size3 = static_cast<HMInputParam<int>*>(InputParams[2])->getVal();
  int omp_scheduling_type = static_cast<HMInputParam<int>*>(InputParams[3])->getVal();
  int omp_chunk_size = static_cast<HMInputParam<int>*>(InputParams[4])->getVal();
  int omp_num_threads = static_cast<HMInputParam<int>*>(InputParams[5])->getVal();
  std::vector<int> loop_ordering = static_cast<HMInputParam<std::vector<int>>*>(InputParams[6])->getVal();

  std::vector<int> default_ordering{0,1,2,3,4};
  int NUM_K = 256;
  std::vector<double> compute_times;

  bool no_sched_init = false;

  if(!initialized) {
    cout << "INITIALIZING" << endl;
    spmv_handler = new SpMV();
    spmv_handler->matrix_name = matrix_name;
    // spmv_handler->NUM_K = NUM_K;
    spmv_handler->initialize_data(1);
    // result = spmv_handler->get_A();
    taco::Tensor<double> temp_result({spmv_handler->NUM_I}, taco::dense);
    initialized = true;
    sparsity = spmv_handler->get_sparsity();
    num_i = spmv_handler->get_num_i();
    num_j = spmv_handler->get_num_j();
    op = "SpMV";

    taco::util::Timer local_timer;
    for(int i = 0; i < 5; i++) { // I used to be equal to 5
        local_timer = spmv_handler->compute_unscheduled(local_timer);
    }

    auto result = local_timer.getResult();
    no_sched_results = result;
    no_sched_time = result.median;
    no_sched_times = result.times;
    no_sched_energy = median(result.energy_consumptions);
    no_sched_energies = result.energy_consumptions;
    no_sched_init = true;
  }

  taco::Tensor<double> temp_result({spmv_handler->NUM_I}, taco::dense);
  compute_times = std::vector<double>();
  
  if(default_config_time == 0.0f) {
    int temp_chunk_size = 16;
    int temp_chunk_size2 = 16;
    int temp_chunk_size3 = 16;
    std::vector<int> temp_loop_ordering{0,1,2,3,4};
    int temp_omp_scheduling_type = 0;
    int temp_omp_chunk_size = 1;
    int temp_omp_num_threads = 32;

    spmv_handler->schedule_and_compute(temp_result, temp_chunk_size, temp_chunk_size2, temp_chunk_size3, 
    temp_loop_ordering, temp_omp_scheduling_type, temp_omp_chunk_size, temp_omp_num_threads, false, 10);
    spmv_handler->set_cold_run();

    default_config_time = spmv_handler->get_compute_time();
    std::cout << spmv_handler->get_num_i() << "," << spmv_handler->get_num_j() << "," << default_config_time << "," << no_sched_time << std::endl;
    logger << spmv_handler->get_num_i() << "," << spmv_handler->get_num_j() << "," << default_config_time << "," << no_sched_time << std::endl;
  }

  bool valid = true;
  if(!no_sched_init) {
    try {
      spmv_handler->schedule_and_compute(temp_result, chunk_size, chunk_size2, chunk_size3, loop_ordering, omp_scheduling_type, omp_chunk_size, omp_num_threads, false, 10);
      spmv_handler->set_cold_run();
    } catch(const taco::TacoException& err) {
      Obj.compute_time = 1000000.0f;
      valid = false;
    }
  }
  Obj.valid = valid;

  if (no_sched_init) {
      Obj.results = no_sched_results;
  } else {
      taco::util::TimeResults local_results = spmv_handler->get_results();
      Obj.results = local_results;
  }
  return Obj;
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveSDDMMDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();
  int unroll_factor = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();
  int omp_scheduling_type = static_cast<HMInputParam<int>*>(InputParams[2])->getVal();
  int omp_chunk_size = static_cast<HMInputParam<int>*>(InputParams[3])->getVal();
  int omp_num_threads = static_cast<HMInputParam<int>*>(InputParams[4])->getVal();
  std::vector<int> loop_ordering = static_cast<HMInputParam<std::vector<int>>*>(InputParams[5])->getVal();
  std::vector<int> default_ordering{0,1,2,3,4};

  for(auto elem : loop_ordering) {
    std::cout << elem << " ";
  }
  std::cout << std::endl;

  //Initialize tensors
  std::vector<double> compute_times;

  bool no_sched_init = false;

  if(!initialized) {
    cout << "INITIALIZING" << endl;
    sddmm_handler = new SDDMM();
    sddmm_handler->matrix_name = matrix_name;
    sddmm_handler->initialize_data(1);
    initialized = true;
    sparsity = sddmm_handler->get_sparsity();
    num_i = sddmm_handler->get_num_i();
    num_j = sddmm_handler->get_num_j();
    // Added for filtering vectors out from suitesparse
    if(num_j == 1 || num_i == 1) {
      exit(1);
    }
    op = "SDDMM";

    // Taco requires you to start with running the deafult
    std::vector<int> tmp_loop_ordering = default_ordering;
    int tmp_chunk_size = 16;
    int tmp_unroll_factor = 8;
    compute_times = vector<double>();
    for(int i = 0; i < 5; i++) {
      double timer = 0.0;
      timer = sddmm_handler->compute_unscheduled();
      compute_times.push_back(timer);
    }
    no_sched_time = median(compute_times);
    no_sched_init = true;
  }


  taco::Tensor<double> temp_result({sddmm_handler->NUM_I, sddmm_handler->NUM_J}, taco::dense);

  if(default_config_time == 0.0f) {
    int temp_chunk_size = 16;
    int temp_unroll_factor = 8;
    std::vector<int> temp_loop_ordering{0,1,2,3,4};
    int temp_omp_scheduling_type = 0;
    int temp_omp_chunk_size = 1;
    int temp_omp_num_threads = 32;
    sddmm_handler->schedule_and_compute(temp_result, temp_chunk_size, temp_unroll_factor, temp_loop_ordering, temp_omp_scheduling_type, temp_omp_chunk_size, temp_omp_num_threads, false, 10);
    sddmm_handler->set_cold_run();

    default_config_time = sddmm_handler->get_compute_time();

    std::cout << sddmm_handler->get_num_i() << "," << sddmm_handler->get_num_j() << "," << default_config_time << "," << no_sched_time << std::endl;
    logger << sddmm_handler->get_num_i() << "," << sddmm_handler->get_num_j() << "," << default_config_time << "," << no_sched_time << std::endl;
  }

  if(!no_sched_init) {
    sddmm_handler->schedule_and_compute(temp_result, chunk_size, unroll_factor, loop_ordering, omp_scheduling_type, omp_chunk_size, omp_num_threads, false, 10);
    sddmm_handler->set_cold_run();
  }

  double compute_time = no_sched_init ? no_sched_time : sddmm_handler->get_compute_time();
  Obj.compute_time = compute_time;
  return Obj;
}

float dummy_result = 200.0f;

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveTTVDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size_i = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();
  int chunk_size_fpos = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();
  int chunk_size_k = static_cast<HMInputParam<int>*>(InputParams[2])->getVal();
  int omp_scheduling_type = static_cast<HMInputParam<int>*>(InputParams[3])->getVal();
  int omp_chunk_size = static_cast<HMInputParam<int>*>(InputParams[4])->getVal();
  int omp_num_threads = static_cast<HMInputParam<int>*>(InputParams[5])->getVal();
  std::vector<int> loop_ordering = static_cast<HMInputParam<std::vector<int>>*>(InputParams[6])->getVal();
  std::vector<int> default_ordering{0,1,2,3,4};

  int NUM_I = 1000;
  int NUM_J = 100;
  int NUM_K = 100;

  std::vector<double> compute_times;

  bool no_sched_init = false;

  if(!initialized) {
    cout << "INITIALIZING" << endl;
    ttv_handler = new TTV();
    ttv_handler->matrix_name = matrix_name;
    ttv_handler->SPARSITY = 0.1;
    ttv_handler->NUM_I = NUM_I;
    ttv_handler->NUM_J = NUM_J;
    ttv_handler->NUM_K = NUM_K;
    ttv_handler->initialize_data(0);
    initialized = true;
    num_i = ttv_handler->NUM_I;
    num_j = ttv_handler->NUM_J;

    // Added for filtering vectors out from suitesparse
    op = "TTV";
    cout << "computing unscheduled" << endl;
    compute_times = vector<double>();
    for(int i = 0; i < 5; i++) {
      double timer = 0.0;
      timer = ttv_handler->compute_unscheduled();
      compute_times.push_back(timer);
    }
    no_sched_time = median(compute_times);
    no_sched_init = true;
    cout << "computed unscheduled" << endl;
  }

  //Initiate scheduling passing in chunk_size (param to optimize)
  bool default_config = (chunk_size_i == 16);
  bool valid = true;
  Obj.valid = valid;

  compute_times = vector<double>();
  ttv_handler->set_cold_run();

  if(default_config_time == 0.0f) {
    std::cout << "NUM_I: " << ttv_handler->NUM_I << ", NUM_J: " << ttv_handler->NUM_J << std::endl;
    taco::Tensor<double> temp_result({ttv_handler->NUM_I, ttv_handler->NUM_J}, taco::dense);
    std::cout << "Computing default unscheduled" << std::endl;
    int temp_chunk_size = 16;
    std::vector<int> temp_loop_ordering{0,1,2,3,4};
    int temp_omp_scheduling_type = 0;
    int temp_omp_chunk_size = 16;
    int temp_chunk_size_i = 16;
    int temp_chunk_size_fpos = 16;
    int temp_chunk_size_k = 16;
    int temp_omp_num_threads = 32;
    ttv_handler->schedule_and_compute(temp_result, temp_chunk_size_i, temp_chunk_size_fpos, temp_chunk_size_k,
                                      temp_loop_ordering, temp_omp_scheduling_type, temp_omp_chunk_size, temp_omp_num_threads, false, 5);
    ttv_handler->set_cold_run();

    default_config_time = ttv_handler->get_compute_time();
    Obj.compute_time = default_config_time;
    logger << ttv_handler->get_num_i() << "," << ttv_handler->get_num_j() << "," << default_config_time << "," << no_sched_time << std::endl;
  }

  if(!no_sched_init) {
    std::cout << "NUM_I: " << ttv_handler->NUM_I << ", NUM_J: " << ttv_handler->NUM_J << std::endl;
    taco::Tensor<double> temp_result({ttv_handler->NUM_I, ttv_handler->NUM_J}, taco::dense);
    try{
  	   ttv_handler->schedule_and_compute(temp_result, chunk_size_i, chunk_size_fpos, chunk_size_k,
                                         loop_ordering, omp_scheduling_type, omp_chunk_size, omp_num_threads, false, 5);
    	 ttv_handler->set_cold_run();
       double compute_time = ttv_handler->get_compute_time();
       Obj.compute_time = compute_time;
	     valid = true;
    } catch (const taco::TacoException& err) {
	     Obj.compute_time = 100000.0f;
  	   valid = false;
    }
    Obj.valid = valid;
  }

  return Obj;
}

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveTTMDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();
  int unroll_factor = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();
  int omp_scheduling_type = static_cast<HMInputParam<int>*>(InputParams[2])->getVal();
  int omp_chunk_size = static_cast<HMInputParam<int>*>(InputParams[3])->getVal();
  int omp_num_threads = static_cast<HMInputParam<int>*>(InputParams[4])->getVal();
  std::vector<int> loop_ordering = static_cast<HMInputParam<std::vector<int>>*>(InputParams[5])->getVal();
  std::vector<int> default_ordering{0,1,2,3,4};

  int NUM_I = 10000;
  int NUM_J = 10000;
  int NUM_K = 1000;

  std::vector<double> compute_times;
  bool valid = true;
  bool no_sched_init = false;

  if(!initialized) {
    cout << "INITIALIZING" << endl;
    ttm_handler = new TTM();
    ttm_handler->matrix_name = matrix_name;
    ttm_handler->NUM_L = 64;
    ttm_handler->initialize_data(1);
    initialized = true;
    num_i = ttm_handler->NUM_I;
    num_j = ttm_handler->NUM_J;
 
    // Added for filtering vectors out from suitesparse
    op = "TTM";

    compute_times = vector<double>();
    for(int i = 0; i < 5; i++) {
      cout << i << endl;
      double timer = 0.0;
      timer = ttm_handler->compute_unscheduled();
      compute_times.push_back(timer);
    }
    Obj.compute_time = median(compute_times);
    no_sched_init = true;
  }
  cout << "a" << endl;
  //Initiate scheduling passing in chunk_size (param to optimize)
  bool default_config = (chunk_size == 16);

  compute_times = vector<double>();
  ttm_handler->set_cold_run();
  taco::Tensor<double> temp_result({ttm_handler->NUM_I, ttm_handler->NUM_J, ttm_handler->NUM_L}, {taco::Sparse, taco::Sparse, taco::Dense});

  std::vector<bool> valid_perm(120, true);
  std::vector<std::vector<int>> orders;
  loop_ordering = vector<int>{0, 1, 2, 3, 4};


  if(default_config_time == 0.0f) {
    std::cout << "Computing default unscheduled" << std::endl;
    int temp_chunk_size = 16;
    int temp_unroll_factor = 8;
    std::vector<int> temp_loop_ordering{0,1,2,3,4};
    int temp_omp_scheduling_type = 0;
    int temp_omp_chunk_size = 1;
    int temp_omp_num_threads = 32;
    ttm_handler->schedule_and_compute(temp_result, temp_chunk_size, temp_unroll_factor, temp_loop_ordering, temp_omp_scheduling_type, temp_omp_chunk_size, temp_omp_num_threads, false);
    ttm_handler->set_cold_run();

    default_config_time = ttm_handler->get_compute_time();
  }

  cout << "post def pre sched" << endl;
  if(!no_sched_init) {
    try{
  	   ttm_handler->schedule_and_compute(temp_result, chunk_size, unroll_factor, loop_ordering, omp_scheduling_type, omp_chunk_size, omp_num_threads, false);
    	 ttm_handler->set_cold_run();
       double compute_time = ttm_handler->get_compute_time();
       Obj.compute_time = compute_time;
	     valid = true;
    } catch (const taco::TacoException& err) {
	     Obj.compute_time = 100000.0f;
  	   valid = false;
    }
    Obj.valid = valid;
  }

  return Obj;
}

HMObjective calculateObjectiveMTTKRPDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger) {
  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();
  int unroll_factor = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();
  int omp_scheduling_type = static_cast<HMInputParam<int>*>(InputParams[2])->getVal();
  int omp_chunk_size = static_cast<HMInputParam<int>*>(InputParams[3])->getVal();
  int omp_num_threads = static_cast<HMInputParam<int>*>(InputParams[4])->getVal();
  std::vector<int> loop_ordering = static_cast<HMInputParam<std::vector<int>>*>(InputParams[5])->getVal();
  std::vector<int> default_ordering{0,1,2,3,4};

  int NUM_I = 10000;
  int NUM_J = 10000;
  int NUM_K = 1000;

  std::vector<double> compute_times;
  bool valid = true;
  bool no_sched_init = false;

  if(!initialized) {
    cout << "INITIALIZING" << endl;
    mttkrp_handler = new MTTKRP();
    mttkrp_handler->matrix_name = matrix_name;
    mttkrp_handler->NUM_J = 2560;
    mttkrp_handler->initialize_data(1);
    initialized = true;
    num_i = mttkrp_handler->NUM_I;
    num_j = mttkrp_handler->NUM_J;
  
    // Added for filtering vectors out from suitesparse
    op = "MTTKRP";

    compute_times = vector<double>();
    for(int i = 0; i < 5; i++) {
      double timer = 0.0;
      timer = mttkrp_handler->compute_unscheduled();
      compute_times.push_back(timer);
    }
    cout << compute_times.size() << endl;
    for (double c : compute_times) {
      cout << c << " " << endl;
    }
    cout << " == " << endl;
    Obj.compute_time = median(compute_times);
    no_sched_init = true;
  }

  //Initiate scheduling passing in chunk_size (param to optimize)
  bool default_config = (chunk_size == 16);

  compute_times = vector<double>();
  taco::Tensor<double> temp_result({mttkrp_handler->NUM_I, mttkrp_handler->NUM_J}, taco::dense);

  if(default_config_time == 0.0f) {
    std::cout << "Computing default unscheduled" << std::endl;
    int temp_chunk_size = 16;
    int temp_unroll_factor = 8;
    std::vector<int> temp_loop_ordering{0,1,2,3,4};
    int temp_omp_scheduling_type = 0;
    int temp_omp_chunk_size = 1;
    int temp_omp_num_threads = 32;
    mttkrp_handler->schedule_and_compute(temp_result, temp_chunk_size, temp_unroll_factor, temp_loop_ordering, temp_omp_scheduling_type, temp_omp_chunk_size, temp_omp_num_threads, false);
    mttkrp_handler->set_cold_run();

    default_config_time = mttkrp_handler->get_compute_time();
  }

  if(!no_sched_init) {
    try{
	     mttkrp_handler->schedule_and_compute(temp_result, chunk_size, unroll_factor, loop_ordering, omp_scheduling_type, omp_chunk_size, omp_num_threads, false);
       mttkrp_handler->set_cold_run();
       double compute_time = mttkrp_handler->get_compute_time();
       Obj.compute_time = compute_time;
	     valid = true;
    } catch (const taco::TacoException& err) {
	     Obj.compute_time = 100000.0f;
  	   valid = false;
    }
    Obj.valid = valid;
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
  if (test_name == "MTTKRP")
    return calculateObjectiveMTTKRPDense(InParams, matrix_name, logger);
  else {
    std::cout << "Test case not implemented yet" << std::endl;
    exit(-1);
  }
}

bool validate_ordering_sddmm(std::vector<int> order) {
  std::unordered_map<int, int> dict;
  for(int i = 0; i < order.size(); i++) {
    dict[order[i]] = i;
    if(i == 1) {
      if(dict.find(1) != dict.end() && dict.find(2) != dict.end()) {
        return false;
      }
    }
    if(i == 2) {
      if(dict.find(1) != dict.end() && dict.find(2) != dict.end() && dict.find(3) != dict.end()) {
        return false;
      }
    }
    if(order[i] == 4) {
      if(dict.find(2) == dict.end() && dict.find(1) == dict.end()) {
        std::cout << "first case triggered\n";
        return false;
      }
      if(dict.find(2) != dict.end()) {
        if(dict.find(0) == dict.end()){ //&& dict.find(1) != dict.end()) {
          std::cout << "second case triggered\n";
          return false;
        }
      }
    }
    if(order[i] == 0) {
      if(dict.find(4) != dict.end()) {
        std::cout << "third case triggered\n";
        return false;
      }
    }
  }
  return true;
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

int single_run_spmm(std::string matrix_name, int chunk_size, int unroll_factor, int omp_scheduling_type, int omp_chunk_size, int omp_num_threads=32) {
  using namespace taco;

  std::vector<int> default_ordering{0,1,2,3,4};
  int NUM_K = 256;

  if(!initialized) {
    cout << "INITIALIZING" << endl;
    spmm_handler = new SpMM();
    spmm_handler->matrix_name = matrix_name;
    spmm_handler->NUM_K = NUM_K;
    spmm_handler->initialize_data(1);
    taco::Tensor<double> temp_result({spmm_handler->NUM_I, spmm_handler->NUM_K}, taco::dense);
    initialized = true;
    sparsity = spmm_handler->get_sparsity();
    num_j = spmm_handler->get_num_j();
    op = "SpMM";
  }

  spmm_handler->set_cold_run();
  taco::Tensor<double> temp_result({spmm_handler->NUM_I, spmm_handler->NUM_K}, taco::dense);
  taco::util::Timer local_timer;
  for(int i = 0; i < 5; i++) {
    local_timer = spmm_handler->compute_unscheduled(local_timer);
   }

  double compute_time = local_timer.getResult().median;

  std::cout << "Compute time: " << compute_time << std::endl;
  return 0;
}


int main(int argc, char **argv) {

  if (!getenv("HYPERMAPPER_HOME")) {
    std::string ErrMsg = "Environment variables are not set!\n";
    ErrMsg += "Please set HYPERMAPPER_HOME before running this ";
    setenv("HYPERMAPPER_HOME", "/home/ubuntu/hypermapper_dev", true);
    printf("Setting HM variable\n");
    // fatalError(ErrMsg);
  }

  // taco::taco_set_num_threads(32);

  argparse::ArgumentParser program("./bin/taco-taco_dse");

  program.add_argument("-e", "--exhaustive")
    .help("Run exhaustive search")
    .default_value(false)
    .implicit_value(true);
  program.add_argument("-d", "--dynamic")
    .help("Set omp scheduling to dynamic")
    .default_value(false)
    .implicit_value(true);
  program.add_argument("-s", "--single_run")
    .help("Run single run of SPMM")
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
    .default_value(20)
    .scan<'i', int>()
    .required();
  program.add_argument("-chunk", "--omp_chunk_size")
    .help("Omp chunk size")
    .default_value(1)
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
  program.add_argument("-t", "--alternative_setting")
    .help("Name extension to the json file.")
    .default_value(std::string(""))
    .required();
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  // std::string test_name = "SpMM";
  std::string test_name, optimization, matrix_name, count, name_extension;
  bool exh_search, single_run, dynamic;
  test_name = program.get<std::string>("--op");
  optimization = program.get<std::string>("--method");
  matrix_name = program.get<std::string>("--matrix_name");
  num_reps = program.get<int>("--num_reps");
  exh_search = program.get<bool>("--exhaustive");
  single_run = program.get<bool>("--single_run");
  dynamic = program.get<bool>("--dynamic");
  count = program.get<std::string>("--count");
  name_extension = program.get<std::string>("--alternative_setting");

  int omp_chunk_size = program.get<int>("--omp_chunk_size");
  bool Predictor = false;
  if (test_name == "TTV" || test_name == "MTTKRP" || test_name == "TTM") {
    Predictor = true;
  }

  std::string log_file_ = "hypermapper_taco_log.csv";
  bool log_exists = fs::exists(log_file_);

  std::ofstream logger_(log_file_, std::ios_base::app);

  if(!log_exists) {
    logger_ << "Op,Size,Chunk size,Time" << std::endl;
  }

  if (single_run) {
    // spmmExhaustiveSearch(matrix_name, logger);
    int chunk_size = 16;
    int unroll_factor = 8;
    int omp_scheduling_type = dynamic ? 1 : 0;
    int omp_num_threads = 32;
    single_run_spmm(matrix_name, chunk_size, unroll_factor, omp_scheduling_type, omp_chunk_size, omp_num_threads);
    // SpMMVarianceTest(logger);
    exit(0);
  }

  // Set these values accordingly

  std::cout << "Matrix: " << matrix_name << std::endl;

  std::string OutputFoldername;
  std::string OutputFoldernameMat;
  std::string ExperimentFolder = "experiments";
  size_t lastindex = matrix_name.find_last_of(".");
  string rawname = matrix_name.substr(0, lastindex);
  OutputFoldernameMat = ExperimentFolder + "/outdata_" + test_name + "_" + rawname;
  OutputFoldername = OutputFoldernameMat + "/" + optimization;



  std::string AppName;
  if (name_extension.length() > 0){
    AppName = "cpp_taco_" + test_name + "_" + name_extension;
  } else {
    AppName = "cpp_taco_" + test_name;
  }
  std::string OutputSubFolderName = AppName + "_" + optimization;

  int dimensionality_plus_one = 7;
  int NumSamples = dimensionality_plus_one;
  int NumIterations = 53;
  std::vector<std::string> Objectives = {"compute_time"};


  // Create output directory if it doesn't exist
  std::string CurrentDir = fs::current_path();
  std::string OutputDir = CurrentDir + "/" + OutputFoldername + "/";
  std::string OutputSubDir = OutputDir + OutputSubFolderName + "/";
  if (fs::exists(OutputDir)) {
    std::cerr << "Output directory exists, continuing!" << std::endl;
    std::string csv_file = OutputFoldername + "/" + AppName + "_" +  optimization + count + "_output_data.csv";
    std::string png_file = OutputFoldername + "/" + test_name + "_plot.png";
    if(fs::exists(csv_file)) {
      std::cerr << "CSV file exists, exiting" << std::endl;
      exit(0);
    }
  } else {

    std::cerr << "Output directory does not exist, creating!" << std::endl;
    if (!fs::create_directories(OutputDir)) {
      fatalError("Unable to create Directory: " + OutputDir);
    }
  }
  if (fs::exists(OutputSubDir)) {
    std::cerr << "Sub Output directory exists, continuing!" << std::endl;
  } else {
    std::cerr << "Sub Output directory does not exist, creating!" << std::endl;
    if (!fs::create_directory(OutputSubDir)) {
      fatalError("Unable to create Directory: " + OutputSubDir);
    }
  }

  std::string log_file = OutputFoldernameMat + "times.csv";
  std::string title_file = OutputFoldernameMat + "title.txt";
  std::string stats_file = OutputFoldernameMat + "stats.txt";
  std::cout << "writing to " << log_file << std::endl;
  bool log_exists_ = fs::exists(log_file);

  std::ofstream logger(log_file, std::ios_base::app);

  if(!log_exists_) {
    logger << "Num_I,Num_J,Default_Time,No_Sched_Time" << std::endl;
  }

  // Collect input parameters
  std::vector<HMInputParamBase *> InParams;

  int numParams = collectInputParams(InParams, test_name);

  const int max_buffer = 1000;
  char buffer[max_buffer];
  std::string JSonFileNameStr;

  // Create json scenario
  JSonFileNameStr =
      createjson(AppName, OutputFoldername + "/" + OutputSubFolderName, NumIterations,
                 NumSamples, InParams, Objectives, Predictor, optimization, count);

  bool grpc = true;
  taco::util::Timer timer;

  if (grpc) {
    timer.start();
  
    std::string server_address("0.0.0.0:50051");
    ConfigurationServiceImpl service(InParams, test_name, matrix_name, logger);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    std::unique_lock<std::mutex> lock(shutdown_mutex);
    shutdown_cv.wait(lock, []{ return shutdown_flag; });  // Wait for shutdown signal

    server->Shutdown();  // Shutdown the server
    server->Wait();  // Optionally wait for all RPC processing to finish
  } else {
    // Launch HyperMapper
    std::string cmd("python ");
    cmd += getenv("HYPERMAPPER_HOME");
    cmd += "/scripts/hypermapper.py";
    // cmd += "/hypermapper/optimizer.py";
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
          std::cout << "Param: " << ParamStr << std::endl;
          fatalError("Unknown parameter received!");
        }
        pos = bufferStr.find_first_of(",\n", pos) + 1;
      }
      for (auto objString : Objectives)
        response += objString + ",";
      if (Predictor) {
        std::cout << response << std::endl;
        response += "Valid";
      }
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
        printHMInputParamBaseVector(InParams);
        HMObjective Obj = calculateObjective(InParams, test_name, matrix_name, logger);  // Function to run hypermapper on
        response += std::to_string(Obj.compute_time);
        if(Predictor){
          response += ",";
          response += to_string(Obj.valid);
        }
        response += "\n";
      }
      std::cout << "Response:\n" << response;
      fputs(response.c_str(), outstream);
      fflush(outstream);
      i++;
    }
    timer.stop();
    cout << "closing pipes" << endl;
    close(hypermapper.from_child);
    close(hypermapper.to_child);
    deleteInputParams(InParams);
    std::cout << "No sched: " << no_sched_time << std::endl;

    cout << JSonFileNameStr << endl;

    std::ofstream logger_title(title_file, std::ios_base::app);
    std::ofstream stats_title(stats_file, std::ios_base::app);

    FILE *fp;
    std::string cmdPareto("python ");
    cmdPareto += getenv("HYPERMAPPER_HOME");
    cmdPareto += "/scripts/plot_optimization_results.py -j";
    cmdPareto += " " + JSonFileNameStr;
    cmdPareto += " -i " + OutputFoldername + " --plot_log -o " + OutputFoldername + "/" + test_name + "_plot.png";
    cmdPareto += " --expert_configuration " + to_string(default_config_time);
    cmdPareto += " -t '" + op + " " + to_string(num_i) + "x" + to_string(num_j) + " d:" + to_string(dimensionality_plus_one - 1) + " sparsity:" + to_string(sparsity) + "'";
    cmdPareto += " -doe ";

    std::string title = op + " " + to_string(num_i) + "x" + to_string(num_j) + " d:" + to_string(dimensionality_plus_one - 1) + " sparsity:" + to_string(sparsity);
    logger_title << title << std::endl;
    stats_title << optimization << "," << timer.getResult().mean << std::endl;
  
    logger_.close();
    logger.close();
    logger_title.close();
    stats_title.close();
  }
  return 0;
}
