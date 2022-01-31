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

#include "taco.h"
#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"

// #include "layer.h"
using namespace taco;

taco::IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");

struct reluAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    // TODO: Figure out necessary iteration algebra
    return Intersect(regions[0], regions[0]);
    // return regions[0];
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

// template<typename T>
// taco::Tensor<T> reshape(taco::Tensor<T> & src, const std::vector<int>& dims) {
  // taco::Tensor<T> dst(dims);
// 
  // if (src.getDimension() != dst.getDimension()) {
    // std::cout << "Incorrect sizes" << std::endl;
  // } else {
    // std::copy_n(src.ptr(), src.size(), dst.ptr());
  // }
// 
  // return dst;
// }


#endif //COMMON_H