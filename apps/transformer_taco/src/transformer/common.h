#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <stdbool.h>
#include <random>
#include <optional>
#include <memory>
#include <functional> // multiplies
#include <numeric> // accumulate

#include "taco.h"
#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"

// #define DEBUG 0

// #include "layer.h"
using namespace taco;

taco::IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
const std::vector<taco::IndexVar> indices{i, j, k, l, m, n};

struct reluAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    return regions[0];
  }
};

struct Relu {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() == 1) << "Operator needs exactly one operand";
    return ir::Max::make(v[0], ir::Literal::zero(v[0].type()));
  }
};

inline taco::Func reluOp() { 
  taco::Func relu("relu", Relu(), reluAlgebra());
  return relu;
}

struct Max {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    if (v.size() == 1) {
      return v[0];
    }
    return ir::Max::make(v);
  }
};


template<typename T>
static IndexExpr maxElem(const Tensor<T>& game, int index) {
  Func op("max", Max());
  if (index == (game.getOrder() - 1)) {
    std::vector<IndexVar> slice;
    for (int i = 0; i <= index; i++) {
      slice.push_back(indices[i]);
    }
    return Reduction(op(), indices[index], game(slice));
  }
  return Reduction(op(), indices[index], maxElem(game, index + 1));
}

template<typename T>
inline void max(taco::Tensor<T> &result, const taco::Tensor<T> &input, int index) {
// inline taco::Tensor<T> max(taco::Tensor<T> &result, taco::Tensor<T> &input, int index) {
  // taco::Tensor<T> result("result", {input.getDimension(0), input.getDimension(1), input.getDimension(2)}, dense);
  result(i, j, k) = maxElem(input, index);
  result.setAssembleWhileCompute(true);
  // result.evaluate();
  result.compile();
  // return result;
}

template<typename T>
inline void printTensor(taco::Tensor<float> t) {
    for(int i = 0; i < t.getDimension(0); ++i) {
        for(int j = 0; j < t.getDimension(1); ++j) { 
            std::cout << t.at({i, j}) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<typename T>
inline void arange(taco::Tensor<T> &in, int low, int high, int step = 1) {
    int counter = low;
    for(int i = low; i < high; ++i) {
        in.insert({i}, (T)counter);
        counter += step;
    } 
}

template<typename T>
void shape(const taco::Tensor<T> &tensor, std::string name) {
        std::cout << name << std::endl;
        for(auto dim : tensor.getDimensions()) {
            std::cout << dim << std::endl;
        }
        std::cout << std::endl;
}

template<typename T>
taco::Tensor<T> reshape(taco::Tensor<T> input, std::vector<int> dimension) {
    int new_dim = dimension.size();
    bool up_size = (new_dim > input.getOrder());
    taco::Tensor<T> output(dimension, dense);
    int dim_product1 = std::accumulate(std::begin(dimension), std::end(dimension), 1, std::multiplies<int>());
    int dim_product2 = std::accumulate(std::begin(input.getDimensions()), std::end(input.getDimensions()), 1, std::multiplies<int>());
    assert(("Product of input dimensions must equal product of reshaped dimensions", dim_product1 == dim_product2));
    for (auto component : input) {
        if(!up_size) {
            int i = component.first[0];
            int j = component.first[1];
            int k = component.first[2];
            int l = component.first[3];
            // Get flatten index: https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays
            int idx = l + input.getDimension(3) * (k + input.getDimension(2) * (j + input.getDimension(1) * i));
            // Get corresponding 3d indices from flatten index
            int new_k = idx % dimension[2];
            int new_j = (idx / dimension[2]) % dimension[1];
            int new_i = idx / (dimension[1] * dimension[2]); //% dimension[0];
            output.insert({new_i, new_j, new_k}, component.second);
        }
        else {
            int i = component.first[0];
            int j = component.first[1];
            int k = component.first[2];
            // Get flatten index: https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays
            int idx = k + input.getDimension(2) * (j + input.getDimension(1) * i);
            // Get corresponding 4d indices from flatten index
            int new_l = idx % dimension[3];
            int new_k = ((idx - new_l) / dimension[3]) % dimension[2];
            int new_j = ((idx - new_k * dimension[3] - new_l) / (dimension[3] * dimension[2])) % dimension[1];
            int new_i = ((idx - new_j * dimension[3] * dimension[2] - new_k * dimension[3] - new_l) / (dimension[3] * dimension[2] * dimension[1])) % dimension[0];
            output.insert({new_i, new_j, new_k, new_l}, component.second);
        }
    }

    output.pack();
    return (output);
}

#endif //COMMON_H