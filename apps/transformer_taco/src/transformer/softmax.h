#ifndef TRANSFORMER_SOFTMAX_H
#define TRANSFORMER_SOFTMAX_H

#include "common.h"
#include "layer.h"

// using namespace taco;

template<typename T>
class Softmax : public Layer<T>{
public:
    Softmax(std::string name = "Softmax") : Layer<T>(name) 
    {

    }
    void forward(taco::Tensor<T> &output, taco::Tensor<T> &input) {
        // taco::Tensor<T> output("out", {input.size(), input[0].size()}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense});
        taco::Tensor<T> exp_sum("exp_sum", {input.getDimensions()[0]}, taco::Format{taco::ModeFormat::Dense});

        exp_sum(i) = exp(input(i, j));
        output(i,j) = exp(input(i, j)) / exp_sum(i);
    }
};

#endif //TRANSFORMER_SOFTMAX_H