#ifndef TRANSFORMER_RELU_H
#define TRANSFORMER_RELU_H

#include "common.h"

// using namespace taco;

template<typename T>
class ReLU : public Layer<T> {
public:
    ReLU() {}
    void forward(taco::Tensor<T> &output, taco::Tensor<T> &input) {
        output(i, j) = input(i, j) * (input(i, j) > 0);

    }
};

#endif //TRANSFORMER_RELU_H