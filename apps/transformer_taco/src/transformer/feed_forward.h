#ifndef TRANSFORMER_FEEDFORWARD_H
#define TRANSFORMER_FEEDFORWARD_H

#include "common.h"
#include "linear.h"
#include "sequential.h"
#include "relu.h"

// using namespace taco;

template<typename T>
class FeedForward {
public:
    FeedForward(int size, int hidden_size) : size(size), hidden_size(hidden_size), sequential(3, hidden_size),
                                             linear_init(size, hidden_size), linear_final(hidden_size, size)
    {
        sequential.add_layer(linear_init);
        sequential.add_layer(ReLU<T>());
        sequential.add_layer(linear_final);
    }
    void forward(taco::Tensor<T> &output, taco::Tensor<T> &input) {
        sequential(output, input);
    }
private:
    Linear<T> linear_init;
    Linear<T> linear_final;
    Sequential<T> sequential;
    int size;
    int hidden_size;
};

#endif //TRANSFORMER_FEEDFORWARD_H