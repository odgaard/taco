#ifndef HPVM_HYPERMAPPER_H
#define HPVM_HYPERMAPPER_H
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>

template <class T> class HMInputParam;
void fatalError(const std::string &msg);

// Enum for HyperMapper parameter types
enum ParamType { Real, Integer, Ordinal, Categorical, Permutation };
enum DataType { Int, Float, IntVector };

std::string getTypeAsString(const ParamType &PT) {
  static const std::map<ParamType, std::string> typeStrings = {
    {Real, "real"},
    {Integer, "integer"},
    {Ordinal, "ordinal"},
    {Categorical, "categorical"},
    {Permutation, "permutation"}
  };
  return typeStrings.at(PT);
}

std::ostream &operator<<(std::ostream &out, const ParamType &PT) {
  static const std::map<ParamType, std::string> typeStrings = {
    {Real, "Real"},
    {Integer, "Integer"},
    {Ordinal, "Ordinal"},
    {Categorical, "Categorical"},
    {Permutation, "Permutation"}
  };
  out << typeStrings.at(PT);
  return out;
}

class HMInputParamBase {
private:
  std::string Name;
  std::string const Key;
  ParamType Type;
  int count = 0;
  DataType DType;

public:
  HMInputParamBase(std::string _Name = "", ParamType _Type = ParamType::Integer,
                   DataType _DType = DataType::Int)
      : Name(_Name), Key(_Name), Type(_Type),
        DType(_DType) {}
  virtual ~HMInputParamBase(){}

  std::string getName() const { return Name; }
  void setName(std::string _Name) { Name = _Name; }

  ParamType getType() const { return Type; }
  void setType(ParamType _Type) { Type = _Type; }

  std::string getKey() const { return Key; }

  DataType getDType() const { return DType; }
  void setDType( DataType _DType) {DType = _DType;}

  bool operator==(const std::string &_Key) {
    return Key == _Key;
  }

  bool operator==(const HMInputParamBase &IP) {
    return Key == IP.getKey();
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

template <typename T>
class LoopReordering {
private:
  std::vector<T> reorder;
  std::vector<std::vector<int>> permutations;
  int num_orderings;

public:
  LoopReordering(std::vector<T> &_reorder)
      : reorder(_reorder), num_orderings(factorial(static_cast<int>(reorder.size()))) {
    compute_permutations();
  }

  ~LoopReordering() {}

  static int factorial(int n) {
    int fact = 1;
    for (int i = 2; i <= n; ++i) {
      fact *= i;
    }
    return fact;
  }

  void compute_permutations() {
    std::vector<int> index_arr(reorder.size());
    std::iota(index_arr.begin(), index_arr.end(), 0);
    permutations.reserve(num_orderings);

    do {
      permutations.push_back(index_arr);
    } while (std::next_permutation(index_arr.begin(), index_arr.end()));
  }

  int get_num_reorderings() const { return num_orderings; }

  std::vector<T> get_reordering(int index) {
    if (index >= num_orderings) {
      std::cerr << "Invalid index entered!" << std::endl;
      exit(1);
    }

    std::vector<T> temp;
    for (int i : permutations[index]) {
      temp.push_back(reorder[i]);
    }
    return temp;
  }

  void get_reordering(std::vector<T>& temp, std::vector<int> reordering) {
    for (int i : reordering) {
      temp.push_back(reorder[i]);
    }
  }

  void print() const {
    for (const auto& perm : permutations) {
      for (int j : perm) {
        std::cout << j << " ";
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
  HMInputParam(std::string _Name = "", ParamType _Type = ParamType::Integer, std::vector<T> _Range = {})
      : HMInputParamBase(_Name, _Type), Range(_Range) {
        if (std::is_same<T, int>::value)
          setDType(Int);
        else if (std::is_same<T, float>::value)
          setDType(Float);
        else if (std::is_same<T, std::vector<int>>::value)
          setDType(IntVector);
        else
          fatalError("Unhandled data type used for input parameter. New data types can be added by augmenting the DataType enum, and modifying this constructor accordingly.");
      }

  ~HMInputParam(){}

  void setRange(std::vector<T> const &_Range) { Range = _Range; }
  std::vector<T> getRange() const { return Range; }

  T getVal() const { return Value; }
  void setVal(T _Value) { Value = _Value; }

  bool operator==(const std::string &_Key) {
    return getKey() == _Key;
  }

  bool operator==(const HMInputParam<T> &IP) {
    return getKey() == IP.getKey();
  }

  template <class U>
  void printRange(std::ostream& out, const std::vector<U>& range, const std::string& prefix = "", const std::string& postfix = "") const {
    out << prefix;
    char separator[1] = "";
    for (const auto& i : range) {
      out << separator << i;
      separator[0] = ',';
    }
    out << postfix;
  }

  void print(std::ostream &out) {
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
    } else if (getType() == ParamType::Permutation) {
      out << "\n  Range: [";
      char separator[1] = "";
      for (std::vector<int> i : getRange()) {
        out << separator << i[0];
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
  bool valid;
};

#endif
