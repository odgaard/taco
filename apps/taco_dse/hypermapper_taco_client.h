#ifndef HPVM_HYPERMAPPER_H
#define HPVM_HYPERMAPPER_H
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

template <class T> class HMInputParam;
void fatalError(const std::string &msg);

// Enum for HyperMapper parameter types
enum ParamType { Real, Integer, Ordinal, Categorical };
enum DataType { Int, Float };

std::ostream &operator<<(std::ostream &out, const ParamType &PT) {
  switch (PT) {
  case Real:
    out << "Real";
    break;
  case Integer:
    out << "Integer";
    break;
  case Ordinal:
    out << "Ordinal";
    break;
  case Categorical:
    out << "Categorical";
    break;
  }
  return out;
}

std::string getTypeAsString(const ParamType &PT) {
  std::string TypeString;
  switch (PT) {
  case Real:
    TypeString = "real";
    break;
  case Integer:
    TypeString = "integer";
    break;
  case Ordinal:
    TypeString = "ordinal";
    break;
  case Categorical:
    TypeString = "categorical";
    break;
  }

  return TypeString;
}

class HMInputParamBase {
private:
  std::string Name;
  std::string const Key;
  ParamType Type;
  inline static int count = 0;
  DataType DType;

public:
  HMInputParamBase(std::string _Name = "", ParamType _Type = ParamType::Integer,
                   DataType _DType = DataType::Int)
      : Name(_Name), Key(_Name), Type(_Type),
        DType(_DType) {}

      // : Name(_Name), Key("x" + std::to_string(count++)), Type(_Type),
  virtual ~HMInputParamBase(){}

  std::string getName() const { return Name; }
  void setName(std::string _Name) { Name = _Name; }

  ParamType getType() const { return Type; }
  void setType(ParamType _Type) { Type = _Type; }

  std::string getKey() const { return Key; }

  DataType getDType() const { return DType; }
  void setDType( DataType _DType) {DType = _DType;}

  bool operator==(const std::string &_Key) {
    if (Key == _Key) {
      return true;
    } else {
      return false;
    }
  }

  bool operator==(const HMInputParamBase &IP) {
    if (Key == IP.getKey()) {
      return true;
    } else {
      return false;
    }
  }

  void print () {
    std::cout << getKey() << ":";
    std::cout << "\n  Name: " << getName();
    std::cout << "\n  Type: " << getType();
    print(std::cout);
  }

  virtual void print(std::ostream &out) const {}

  friend std::ostream &operator<<(std::ostream &out,
                                  const HMInputParamBase &IP) {
    out << IP.getKey() << ":";
    out << "\n  Name: " << IP.getName();
    out << "\n  Type: " << IP.getType();
    IP.print(out);
    return out;
  }
};

inline int factorial(int n) {
    int fact = 1;
    for (int i = 2; i <= n; i++)
        fact = fact * i;
    return fact;
}

template <class T> class LoopReordering {
private:
  std::vector<T> reorder;
  // std::vector< std::vector<T> > permutations;
  // T** permutations;
  int **permutations;
  int *index_arr;
  int num_orderings;
public:
  // TODO: Can't seem to work for IndexVar directly
  LoopReordering(std::vector<T> &_reorder)
    : reorder(_reorder), num_orderings(factorial(reorder.size())) {
    // permutations = new T*[num_orderings];
    permutations = new int*[num_orderings];
    index_arr = new int[num_orderings];
    for(int i = 0; i < num_orderings; i++) {
      // permutations[i] = new T[reorder.size()];
      permutations[i] = new int[reorder.size()];
      index_arr[i] = i;
    }
    // std::cout << "num_orders: " << num_orderings << std::endl;
  }
  ~LoopReordering() {
    // TODO: Figure out how to properly clear memory
    delete [] index_arr;
    for(int i = 0; i < num_orderings; i++) {
      if(permutations[i])
        delete [] permutations[i];
    }
    if(permutations)
      delete [] permutations;
  }
  
  void compute_permutations() {
    int count = 0;
    int reorder_size = reorder.size();
    std::sort(index_arr, index_arr + reorder_size);
    do {
      memcpy((void *)permutations[count], (void *)index_arr, reorder_size * sizeof(int));
      count++;
    } while(std::next_permutation(index_arr, index_arr + reorder_size));
  }
  
  int get_num_reorderings() { return num_orderings; }

  std::vector<T> get_reordering(int index) {
    if(index > num_orderings) {
      std::cerr << "Invalid index entered!" << std::endl;
      exit(1);
    }
    std::vector<T> temp;

    int size = static_cast<int>(reorder.size());
    int index_ptr;
    for(int i = 0; i < size; i++){
      index_ptr = permutations[index][i];
      temp.push_back(reorder[index_ptr]);
    }
    return temp;
  }

  void print() {
    int size = static_cast<int>(reorder.size());
    for (int i = 0; i < num_orderings; i++) {
      for (int j = 0; j < size; j++) {
        std::cout << permutations[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }
};

// HyperMapper Input Parameter object
template <class T> class HMInputParam : public HMInputParamBase {
private:
  std::vector<T> Range;
  T Value;

public:
  HMInputParam(std::string _Name = "", ParamType _Type = ParamType::Integer)
      : HMInputParamBase(_Name, _Type) {
        if (std::is_same<T, int>::value)
          setDType(Int);
        else if (std::is_same<T, float>::value)
          setDType(Float);
        else
          fatalError("Unhandled data type used for input parameter. New data types can be added by augmenting the DataType enum, and modifying this constructor accordingly.");
      }

  ~HMInputParam(){}

  void setRange(std::vector<T> const &_Range) { Range = _Range; }
  std::vector<T> getRange() const { return Range; }

  T getVal() const { return Value; }
  void setVal(T _Value) { Value = _Value; }

  bool operator==(const std::string &_Key) {
    if (getKey() == _Key) {
      return true;
    } else {
      return false;
    }
  }

  bool operator==(const HMInputParam<T> &IP) {
    if (getKey() == IP.getKey()) {
      return true;
    } else {
      return false;
    }
  }

  void print(std::ostream &out) const {
    if (getType() == ParamType::Ordinal ||
        getType() == ParamType::Categorical) {
      out << "\n  Range: {";
      char separator[1] = "";
      for (auto i : getRange()) {
        out << separator << i;
        separator[0] = ',';
      }
      out << "}";
    } else if (getType() == ParamType::Integer ||
               getType() == ParamType::Real) {
      out << "\n  Range: [";
      char separator[1] = "";
      for (auto i : getRange()) {
        out << separator << i;
        separator[0] = ',';
      }
      out << "]";
    }
  }

  friend std::ostream &operator<<(std::ostream &out,
                                  const HMInputParam<T> &IP) {
    out << IP.getKey() << ":";
    out << "\n  Name: " << IP.getName();
    out << "\n  Type: " << IP.getType();
    IP.print(out);
    return out;
  }
};

// HyperMapper Objective object
struct HMObjective {
  float compute_time;
};

#endif
