#ifndef TRANSFORMER_LINEAR_H
#define TRANSFORMER_LINEAR_H

#include "common.h"

// using namespace taco;

template<typename T>
class Linear : public Layer<T> {
public:
    Linear(int in_size, int out_size, std::optional<taco::Tensor<T>> bias_init, bool with_bias = false, std::string name = "Linear") : Layer(name), in_size(in_size), out_size(out_size),  with_bias(with_bias), 
                                                                                     weights("weights", {in_size, out_size}, taco::Format{taco::Mode::Dense, taco::Mode::Dense}), 
                                                                                     bias(with_bias ? taco::TensorBase(bias_init) : default) 
    {

    }
    void forward(taco::Tensor<T> &output, taco::Tensor<T> &input) {
        // Dot product of input and weights transposed
        if(!weights_initialized) { 
            initialize_weights();
        }
        output(i) = input(i) * weights(j, i);

        // Add bias if any, likely won't need it
        if(with_bias) {
            output(i) += bias(i);
        }
    }

    void initialize_weights() {
        std::default_random_engine seed(time(0));
        std::normal_distribution<T> dist(0, 1);

        // Initialize weights using normal distribution with mean 0, stddev 1 and
        // multiply sample with sqrt of (2 / n) n -> input size 
        for (int i = 0; i <in_size; ++i) { 
            for(int j = 0; j < out_size; ++j) {
                weights.insert({i, j}, (T)dist(seed) * std::sqrt(2 / in_size));
            }
        }
        weights_initialized = true;
    }

private:
    taco::Tensor<T> weights;
    taco::Tensor<T> bias;
    int in_size;
    int out_size;
    bool with_bias;
    bool weights_initialized; // might not be needed, could always generate instead of passing in
    bool bias_initialized; // Same as above
};

#endif //TRANSFORMER_LINEAR_H