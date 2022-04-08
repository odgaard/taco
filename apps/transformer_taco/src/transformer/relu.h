#ifndef TRANSFORMER_RELU_H
#define TRANSFORMER_RELU_H

#include "common.h"
#include "layer.h"

// using namespace taco;

template<typename T>
class ReLU : public Layer<T> {
public:
    // ReLU() {}
    ReLU(std::string name = "Relu") : Layer<T>(name) {}
    void forward(taco::Tensor<T> &output, const taco::Tensor<T> &input) {
        // output(i, j) = input(i, j) * (input(i, j) > 0);
        Func relu = reluOp();
        output(i, j, k) = relu(input(i, j, k));

    }
};

#endif //TRANSFORMER_RELU_H