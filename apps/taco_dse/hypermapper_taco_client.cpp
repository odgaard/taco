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

#include <dirent.h>
#include <sys/stat.h>
#include <set>
#include <cstdlib>

#include <algorithm>
#include <cctype>

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

bool energy = false;

namespace fs = std::experimental::filesystem;
int WARP_SIZE = 32;
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

std::unordered_map<std::string,bool> conf_cache;
std::condition_variable shutdown_cv;
std::mutex shutdown_mutex;
bool shutdown_flag = false;

void fatalError(const std::string &msg);
void addCommonParams(std::vector<HMInputParamBase *> &InParams);
int collectInputParamsSpMV(std::vector<HMInputParamBase *> &InParams, int SPLIT);
int collectInputParamsSpMM(std::vector<HMInputParamBase *> &InParams);
int collectInputParamsSDDMM(std::vector<HMInputParamBase *> &InParams);
int collectInputParamsTTV(std::vector<HMInputParamBase *> &InParams);
int collectInputParamsTTM(std::vector<HMInputParamBase *> &InParams);
int collectInputParamsMTTKRP(std::vector<HMInputParamBase *> &InParams);
int collectInputParams(std::vector<HMInputParamBase *> &InParams, std::string test_name);
std::vector<HMInputParamBase *>::iterator findHMParamByKey(std::vector<HMInputParamBase *> &InParams, const std::string& Key);
HMObjective calculateObjectiveSpMVDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger);
HMObjective calculateObjectiveSpMMDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger);
HMObjective calculateObjectiveSDDMMDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger);
HMObjective calculateObjectiveTTVDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger);
HMObjective calculateObjectiveTTMDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger);
HMObjective calculateObjectiveMTTKRPDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger);
HMObjective calculateObjective(std::vector<HMInputParamBase *> &InParams, std::string test_name, std::string matrix_name, std::ofstream &logger);
std::string createjson(std::string AppName, std::string OutputFoldername, int NumIterations,
                  int NumDSERandomSamples, std::vector<HMInputParamBase *> &InParams,
                  std::vector<std::string> Objectives, bool predictor, std::string optimization, std::string count);
double median(vector<double> vec);
bool validate_ordering_sddmm(std::vector<int> order);
bool validate_ordering(std::vector<int> order);


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
    }


bool startsWith(const std::string &mainStr, const std::string &toMatch) {
    if (mainStr.size() >= toMatch.size() &&
        mainStr.compare(0, toMatch.size(), toMatch) == 0)
        return true;
    else
        return false;
}

void removeContentsOfDirectoriesMatchingPattern(const std::string& directory, const std::string& pattern) {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directory.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string dirName = ent->d_name;
            if (startsWith(dirName, pattern)) {
                std::string fullPath = directory + "/" + dirName;
                struct stat statbuf;
                if (stat(fullPath.c_str(), &statbuf) == 0 && S_ISDIR(statbuf.st_mode)) {
                    // Use system call to remove the contents of the directory
                    std::string removeCommand = "rm -rf " + fullPath + "/*";
                    system(removeCommand.c_str());
                    std::cout << "Cleared contents of directory: " << fullPath << std::endl;
                }
            }
        }
        closedir(dir);
    } else {
        // Could not open directory
        perror("Could not open directory");
    }
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
        int result_code = gethostname(hostname, HOST_NAME_MAX);
        if (result_code != 0) {
          // Handle error or log it
          std::cerr << "Error getting hostname" << std::endl;
        }

        // Access and process the configurations:
        const Configuration& config = request->configurations();
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
            } else if (parameter.has_real_param()) {
                setParameterValue(hmParam, parameter.real_param().value());
            } else if (parameter.has_categorical_param()) {
                setParameterValue(hmParam, parameter.categorical_param().value());
            } else if (parameter.has_ordinal_param()) {
                setParameterValue(hmParam, parameter.ordinal_param().value());
            } else if (parameter.has_string_param()) {
                setParameterValue(hmParam, parameter.string_param().value());
            } else if (parameter.has_permutation_param()) {
                setParameterValue(hmParam, parameter.permutation_param().values());
            } else {
              return Status::CANCELLED;
            }
        }

        HMObjective obj;
        std::vector<double> temp_meds;
        std::vector<double> temp_energy_consumptions;
        std::vector<double> all_times;
        std::vector<double> all_energy;
        double temp_med;
        double temp_energy;
        bool feasible_bool = true;
        int iterations = 5;
        for (int i = 0; i < iterations; i++) {
          try {
              obj = calculateObjective(m_InParams, m_test_name, m_matrix_name, m_logger);
              if(energy) {
                temp_energy = med(obj.results.energy_consumptions);
                all_energy.insert(all_energy.end(), obj.results.energy_consumptions.begin(), obj.results.energy_consumptions.end());
                temp_energy_consumptions.push_back(temp_energy);
              } else {
                all_energy = std::vector<double>(iterations, 0.0f);
                temp_energy_consumptions = std::vector<double>(iterations, 0.0f);
              }
              temp_med = med(obj.results.times);
              all_times.insert(all_times.end(), obj.results.times.begin(), obj.results.times.end());
              temp_meds.push_back(temp_med);
              if (obj.valid == false) {
                feasible_bool = false;
              }
          //} catch (const std::exception& e) {
          } catch(const taco::TacoException& e) {
              m_logger << "Exception caught: " << e.what() << std::endl;
              feasible_bool = false;
              temp_meds = std::vector<double>(iterations, 0.0f);
              temp_energy_consumptions = std::vector<double>(iterations, 0.0f);
              all_times = std::vector<double>(iterations, 0.0f);
              all_energy = std::vector<double>(iterations, 0.0f);
              Feasible feasible;
              feasible.set_value(feasible_bool); // Mocked feasibility value
              response->mutable_feasible()->CopyFrom(feasible);
              return Status(StatusCode::INTERNAL, std::string("TACO Server error: ") + e.what());
          } catch (const std::runtime_error& e) {
              m_logger << "Runtime error caught: " << e.what() << std::endl;
              return Status(StatusCode::INTERNAL, std::string("Runtime error: ") + e.what());
          } catch (const std::exception& e) {
              m_logger << "Standard exception caught: " << e.what() << std::endl;
              return Status(StatusCode::INTERNAL, std::string("General Server error: ") + e.what());
          } catch (...) {
              m_logger << "Unknown exception caught" << std::endl;
              return Status(StatusCode::INTERNAL, "Unknown error");
          }
          std::cout << temp_med << std::endl;
        }
        std::cout << "Median: " << med(temp_meds) << std::endl;
        double new_med = med(temp_meds);
        double new_energy = 0.0;
        if (energy) {
          std::cout << "Energy: " << med(temp_energy_consumptions) << std::endl;
          double new_energy = med(temp_energy_consumptions);
        }

        std::this_thread::sleep_for(std::chrono::seconds(10));
        //removeContentsOfDirectoriesMatchingPattern("/tmp", "taco_");

        // Create a mocked response:
        taco::util::TimeResults local_results = obj.results;
        std::cout << "[" << hostname << "]: " << new_med << ", " << new_energy << std::endl;
        
        Metric metric_median_time;
        metric_median_time.set_name("compute_time");
        metric_median_time.add_values(new_med);
        response->add_metrics()->CopyFrom(metric_median_time);

        Metric metric_compute_times;
        //for (double time : local_results.times) {
        metric_compute_times.set_name("compute_times");
        for (double time : all_times) {
            metric_compute_times.mutable_values()->Add(time);
        }
        response->add_metrics()->CopyFrom(metric_compute_times);

        Metric metric_median_energy;
        metric_median_energy.set_name("energy");
        metric_median_energy.add_values(new_energy);
        response->add_metrics()->CopyFrom(metric_median_energy);

        Metric metric_energy_consumptions;
        metric_energy_consumptions.set_name("energy_consumptions");
        for (double energy_consumption : all_energy) {
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
        feasible.set_value(feasible_bool); // Mocked feasibility value
        response->mutable_feasible()->CopyFrom(feasible);

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
  InParams.push_back(new HMInputParam<int>("omp_scheduling_type", ParamType::Ordinal, {0, 1, 2}));
  InParams.push_back(new HMInputParam<int>("omp_monotonic", ParamType::Ordinal, {0, 1}));
  InParams.push_back(new HMInputParam<int>("omp_dynamic", ParamType::Ordinal, {0, 1}));
  InParams.push_back(new HMInputParam<int>("omp_proc_bind", ParamType::Ordinal, {0, 1, 2}));
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
  if (test_name == "spmv")
    return collectInputParamsSpMV(InParams);
  if (test_name == "spmvsparse")
    return collectInputParamsSpMV(InParams, 1);
  if (test_name == "spmm")
    return collectInputParamsSpMM(InParams);
  if (test_name == "sddmm")
    return collectInputParamsSDDMM(InParams);
  if (test_name == "ttv")
    return collectInputParamsTTV(InParams);
  if (test_name == "ttm")
    return collectInputParamsTTM(InParams);
  if (test_name == "mttkrp")
    return collectInputParamsMTTKRP(InParams);
  else {
    std::cout << "Test case not implemented yet " << test_name << std::endl;
    exit(-1);
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

double median(vector<double> vec) {
        typedef vector<int>::size_type vec_sz;

        vec_sz size = vec.size();
        if (size == 0)
            throw domain_error("median of an empty vector");

        sort(vec.begin(), vec.end());

        vec_sz mid = size/2;

        return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
}


#include <set>
#include <vector>
#include <tuple>
#include <cstdlib>

struct ConfigurationStruct {
    int chunk_size;
    int unroll_factor;
    int omp_scheduling_type;
    std::vector<int> loop_ordering;

    // Constructor
    ConfigurationStruct(int cs, int uf, int ost, const std::vector<int>& lo) :
        chunk_size(cs), unroll_factor(uf), omp_scheduling_type(ost), loop_ordering(lo) {}

    bool operator<(const ConfigurationStruct& other) const {
        if (std::tie(chunk_size, unroll_factor, omp_scheduling_type) != 
            std::tie(other.chunk_size, other.unroll_factor, other.omp_scheduling_type)) {
            return std::tie(chunk_size, unroll_factor, omp_scheduling_type) <
                   std::tie(other.chunk_size, other.unroll_factor, other.omp_scheduling_type);
        }
        return loop_ordering < other.loop_ordering;
    }
};


std::map<ConfigurationStruct, int> seen_configurations;
int next_id = 1; // Start with an ID of 1

// Function that takes input parameters and generates objective
HMObjective calculateObjectiveSpMMDense(std::vector<HMInputParamBase *> &InputParams, std::string matrix_name, std::ofstream &logger) {

  using namespace taco;
  HMObjective Obj;
  int chunk_size = static_cast<HMInputParam<int>*>(InputParams[0])->getVal();
  int unroll_factor = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();
  int omp_chunk_size = static_cast<HMInputParam<int>*>(InputParams[2])->getVal();
  int omp_num_threads = static_cast<HMInputParam<int>*>(InputParams[3])->getVal();
  int omp_scheduling_type = static_cast<HMInputParam<int>*>(InputParams[4])->getVal();
  int omp_monotonic = static_cast<HMInputParam<int>*>(InputParams[5])->getVal();
  int omp_dynamic = static_cast<HMInputParam<int>*>(InputParams[6])->getVal();
  int omp_proc_bind = static_cast<HMInputParam<int>*>(InputParams[7])->getVal();
  std::vector<int> loop_ordering = static_cast<HMInputParam<std::vector<int>>*>(InputParams[8])->getVal();
  
  std::vector<int> default_ordering{0,1,2,3,4};
  // int NUM_I = 67173;
  // int NUM_J = 67173;
  int NUM_K = 256;
  int ITERATIONS = 10; // Was 10
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
    if (energy) {
      no_sched_energy = median(result.energy_consumptions);
      no_sched_energies = result.energy_consumptions;
    }
    no_sched_init = true;
  }

  compute_times = std::vector<double>();
  spmm_handler->set_cold_run();
  taco::Tensor<double> temp_result({spmm_handler->NUM_I, spmm_handler->NUM_K}, taco::dense);

  if(!no_sched_init) {
    spmm_handler->schedule_and_compute(temp_result,
      chunk_size, unroll_factor, loop_ordering,
      omp_scheduling_type, omp_chunk_size,
      omp_monotonic, omp_dynamic, omp_num_threads,
      false, ITERATIONS
    );
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

  int omp_chunk_size = static_cast<HMInputParam<int>*>(InputParams[3])->getVal();
  int omp_num_threads = static_cast<HMInputParam<int>*>(InputParams[4])->getVal();
  int omp_scheduling_type = static_cast<HMInputParam<int>*>(InputParams[5])->getVal();
  int omp_monotonic = static_cast<HMInputParam<int>*>(InputParams[6])->getVal();
  int omp_dynamic = static_cast<HMInputParam<int>*>(InputParams[7])->getVal();
  int omp_proc_bind = static_cast<HMInputParam<int>*>(InputParams[8])->getVal();
  std::vector<int> loop_ordering = static_cast<HMInputParam<std::vector<int>>*>(InputParams[9])->getVal();

  std::vector<int> default_ordering{0,1,2,3,4};
  std::vector<double> compute_times;

  bool no_sched_init = false;

  if(!initialized) {
    cout << "INITIALIZING" << endl;
    spmv_handler = new SpMV();
    spmv_handler->matrix_name = matrix_name;
    spmv_handler->initialize_data(1);
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
    if (energy) {
      no_sched_energy = median(result.energy_consumptions);
      no_sched_energies = result.energy_consumptions;
    }
    no_sched_init = true;
  }

  taco::Tensor<double> temp_result({spmv_handler->NUM_I}, taco::dense);
  compute_times = std::vector<double>();

  bool valid = true;
  if(!no_sched_init) {
    try {
      spmv_handler->schedule_and_compute(temp_result, chunk_size, chunk_size2, chunk_size3, loop_ordering, omp_scheduling_type, omp_chunk_size, omp_monotonic, omp_dynamic, omp_num_threads, false, 10);
      spmv_handler->set_cold_run();
    } catch(const taco::TacoException& e) {
      Obj.compute_time = 1000000.0f;
      valid = false;
      std::cout << "Exception caught: " << e.what() << std::endl;
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

  int omp_chunk_size = static_cast<HMInputParam<int>*>(InputParams[2])->getVal();
  int omp_num_threads = static_cast<HMInputParam<int>*>(InputParams[3])->getVal();
  int omp_scheduling_type = static_cast<HMInputParam<int>*>(InputParams[4])->getVal();
  int omp_monotonic = static_cast<HMInputParam<int>*>(InputParams[5])->getVal();
  int omp_dynamic = static_cast<HMInputParam<int>*>(InputParams[6])->getVal();
  int omp_proc_bind = static_cast<HMInputParam<int>*>(InputParams[7])->getVal();
  std::vector<int> loop_ordering = static_cast<HMInputParam<std::vector<int>>*>(InputParams[8])->getVal();
  std::vector<int> default_ordering{0,1,2,3,4};

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
    compute_times = vector<double>();
    taco::util::Timer local_timer;
    for(int i = 0; i < 5; i++) {
      double timer = 0.0;
      local_timer = sddmm_handler->compute_unscheduled(local_timer);
    }
    auto result = local_timer.getResult();
    no_sched_results = result;
    no_sched_time = result.median;
    no_sched_times = result.times;
    if (energy) {
      no_sched_energy = median(result.energy_consumptions);
      no_sched_energies = result.energy_consumptions;
    }
    no_sched_init = true;
  }

  taco::Tensor<double> temp_result({sddmm_handler->NUM_I, sddmm_handler->NUM_J}, taco::dense);

  if(!no_sched_init) {
    sddmm_handler->schedule_and_compute(temp_result, chunk_size, unroll_factor, loop_ordering, omp_scheduling_type, omp_chunk_size, omp_monotonic, omp_dynamic, omp_proc_bind, omp_num_threads, false, 10);
    sddmm_handler->set_cold_run();
  }

  if (no_sched_init) {
      Obj.results = no_sched_results;
  } else {
      taco::util::TimeResults local_results = sddmm_handler->get_results();
      Obj.results = local_results;
  }
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
  int omp_chunk_size = static_cast<HMInputParam<int>*>(InputParams[3])->getVal();
  int omp_num_threads = static_cast<HMInputParam<int>*>(InputParams[4])->getVal();
  int omp_scheduling_type = static_cast<HMInputParam<int>*>(InputParams[5])->getVal();
  int omp_monotonic = static_cast<HMInputParam<int>*>(InputParams[6])->getVal();
  int omp_dynamic = static_cast<HMInputParam<int>*>(InputParams[7])->getVal();
  int omp_proc_bind = static_cast<HMInputParam<int>*>(InputParams[8])->getVal();
  std::vector<int> loop_ordering = static_cast<HMInputParam<std::vector<int>>*>(InputParams[9])->getVal();
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
  }

  //Initiate scheduling passing in chunk_size (param to optimize)
  bool valid = true;

  ttv_handler->set_cold_run();

  taco::Tensor<double> temp_result({ttv_handler->NUM_I, ttv_handler->NUM_J}, taco::dense);
  try {
    ttv_handler->schedule_and_compute(temp_result, chunk_size_i, chunk_size_fpos, chunk_size_k,
                                       loop_ordering, omp_scheduling_type, omp_chunk_size, 
                                       omp_monotonic, omp_dynamic, omp_proc_bind, omp_num_threads, false, 5);
  	ttv_handler->set_cold_run();
  } catch (const taco::TacoException& err) {
    valid = false;
  }
  Obj.valid = valid;

/*  if (no_sched_init) {
      Obj.results = no_sched_results;
  } else {
    */
      taco::util::TimeResults local_results = ttv_handler->get_results();
      Obj.results = local_results;
  //}
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

  compute_times = vector<double>();
  ttm_handler->set_cold_run();
  taco::Tensor<double> temp_result({ttm_handler->NUM_I, ttm_handler->NUM_J, ttm_handler->NUM_L}, {taco::Sparse, taco::Sparse, taco::Dense});

  std::vector<bool> valid_perm(120, true);
  std::vector<std::vector<int>> orders;
  loop_ordering = vector<int>{0, 1, 2, 3, 4};

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
  int omp_chunk_size = static_cast<HMInputParam<int>*>(InputParams[2])->getVal();
  int omp_num_threads = static_cast<HMInputParam<int>*>(InputParams[3])->getVal();
  int omp_scheduling_type = static_cast<HMInputParam<int>*>(InputParams[4])->getVal();
  int omp_monotonic = static_cast<HMInputParam<int>*>(InputParams[5])->getVal();
  int omp_dynamic = static_cast<HMInputParam<int>*>(InputParams[6])->getVal();
  int omp_proc_bind = static_cast<HMInputParam<int>*>(InputParams[7])->getVal();
  std::vector<int> loop_ordering = static_cast<HMInputParam<std::vector<int>>*>(InputParams[8])->getVal();
  std::vector<int> default_ordering{0,1,2,3,4};

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
    no_sched_init = true;
  }

  compute_times = vector<double>();
  taco::Tensor<double> temp_result({mttkrp_handler->NUM_I, mttkrp_handler->NUM_J}, taco::dense);

  try {
	   mttkrp_handler->schedule_and_compute(temp_result, chunk_size, unroll_factor, loop_ordering, omp_scheduling_type, omp_chunk_size, omp_monotonic, omp_dynamic, omp_proc_bind, omp_num_threads, false);
     mttkrp_handler->set_cold_run();
     taco::util::TimeResults local_results = mttkrp_handler->get_results();
     Obj.results = local_results;
	   valid = true;
  } catch (const taco::TacoException& err) {
	   Obj.compute_time = 100000.0f;
    valid = false;
  }
  Obj.valid = valid;

  return Obj;
}

HMObjective calculateObjective(std::vector<HMInputParamBase *> &InParams, std::string test_name, std::string matrix_name, std::ofstream &logger) {
  if (test_name == "spmv")
    return calculateObjectiveSpMVDense(InParams, matrix_name, logger);
  if (test_name == "spmm")
    return calculateObjectiveSpMMDense(InParams, matrix_name, logger);
  if (test_name == "sddmm")
    return calculateObjectiveSDDMMDense(InParams, matrix_name, logger);
  if (test_name == "ttv")
    return calculateObjectiveTTVDense(InParams, matrix_name, logger);
  if (test_name == "ttm")
    return calculateObjectiveTTMDense(InParams, matrix_name, logger);
  if (test_name == "mttkrp")
    return calculateObjectiveMTTKRPDense(InParams, matrix_name, logger);
  else {
    std::cout << "Test case not implemented yet" << std::endl;
    exit(-1);
  }
}

bool validate_ordering_sddmm(std::vector<int> order) {
  std::unordered_map<int, int> dict;
  for(size_t i = 0; i < order.size(); i++) {
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
  for(long unsigned int i = 0; i < order.size(); i++) {
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

int main(int argc, char **argv) {
  //setenv("OMP_PROC_BIND", omp_proc_bind, 1);
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
  test_name = program.get<std::string>("--op");
  //optimization = program.get<std::string>("--method");
  optimization = "random";
  //matrix_name = program.get<std::string>("--matrix_name");
  std::transform(test_name.begin(), test_name.end(), test_name.begin(),
    [](unsigned char c){ return std::tolower(c); });
  if (test_name == "spmm") {
    matrix_name = "cage12/cage12.mtx";
  } else if (test_name == "spmv") {
    matrix_name = "cage13/cage13.mtx";
  } else if (test_name == "sddmm") {
    matrix_name = "cage10/cage10.mtx";
  } else if (test_name == "ttv") {
    matrix_name = "uber-pickups.tns";
  } else if (test_name == "mttkrp") {
    matrix_name = "facebook.tns";
  }
  num_reps = program.get<int>("--num_reps");
  count = program.get<std::string>("--count");
  name_extension = program.get<std::string>("--alternative_setting");

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

  collectInputParams(InParams, test_name);

  std::string JSonFileNameStr;

  // Create json scenario
  JSonFileNameStr =
      createjson(AppName, OutputFoldername + "/" + OutputSubFolderName, NumIterations,
                 NumSamples, InParams, Objectives, Predictor, optimization, count);

  taco::util::Timer timer;

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
  return 0;
}
