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
#include "layer.h"
// using namespace taco;

const taco::IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");

template<typename T>
taco::Tensor<T> reshape(taco::Tensor<T> & src, const std::vector<int>& dims) {
  taco::Tensor<T> dst(dims);

  if (src.getDimension() != dst.getDimension()) {
    std::cout << "Incorrect sizes" << std::endl;
  } else {
    std::copy_n(src.ptr(), src.size(), dst.ptr());
  }

  return dst;
}


#endif //COMMON_H