#ifndef TRANSFORMER_FEEDFORWARD_H
#define TRANSFORMER_FEEDFORWARD_H

#include "common.h"
#include "layer.h"
#include "linear.h"
#include "sequential.h"
#include "relu.h"
#include "dropout.h"

// using namespace taco;

template<typename T>
class FeedForward {
public:
    FeedForward(int size, int hidden_size, float dropout_prob) : dropout_prob(dropout_prob), size(size), 
                                                                 hidden_size(hidden_size), 
                                                                 sequential(hidden_size),
                                                                 linear_init(size, hidden_size), linear_final(hidden_size, size),
                                                                 relu(), dropout(dropout_prob)
    {
        // Linear<T> *linear1 = new Linear<T>(size, hidden_size);
        // sequential.add_layer(linear1);
        // sequential.add_layer(new ReLU<T>());
        // sequential.add_layer(new Linear<T>(hidden_size, size));
        // sequential.add_layer(new Dropout<T>(dropout_prob));
        // TODO: Take pointer out
        // linear_init = new Linear<T>(size, hidden_size);
        // relu = new ReLU<T>();
        // linear_final = new Linear<T>(hidden_size, size);
        // dropout = new Dropout<T>(dropout_prob);
        // sequential.add_layer()
    }
    ~FeedForward() {
        // delete linear_init;
        // delete relu;
        // delete linear_final;
        // delete dropout;
    }
    void print_weights() {
        std::cout << linear_init.get_weights() << std::endl;
    }
    void forward(taco::Tensor<T> &output, taco::Tensor<T> &input) {
        // std::cout << "Calling sequential";
        // sequential(output, input);
        int batch_size = input.getDimension(0);
        taco::Tensor<T> tmp1("tmp1", {batch_size, size, hidden_size}, dense);
        taco::Tensor<T> tmp2("tmp2", {batch_size, size, hidden_size}, dense);

        linear_init.forward(tmp1, input);
        relu.forward(tmp2, tmp1);
        linear_final.forward(output, tmp2);
        output = dropout.forward(output);
    }
private:
    float dropout_prob;
    int size;
    int hidden_size;
    Sequential<T> sequential;
    Linear<T> linear_init;
    Linear<T> linear_final;
    ReLU<T> relu;
    Dropout<T> dropout;
};

#endif //TRANSFORMER_FEEDFORWARD_H