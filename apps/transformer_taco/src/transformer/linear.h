#ifndef TRANSFORMER_LINEAR_H
#define TRANSFORMER_LINEAR_H

#include "common.h"
#include "layer.h"

// using namespace taco;

template<typename T>
class Linear : public Layer<T> {
public:
    Linear(std::string name = "Linear") : Layer<T>(name) {} 
    Linear(int in_size, int out_size, bool with_bias = false, std::string name = "Linear") : Layer<T>(name), 
                                                                                     in_size(in_size), out_size(out_size),  with_bias(with_bias), 
                                                                                     weights("weights", {in_size, out_size}, {dense, dense})
    {
        initialize_weights();
        // weights_transposed = weights.transpose({1, 0}) ;
    }
    void linear_1d(taco::Tensor<T> &output, const taco::Tensor<T> &input) {
        output(j) = weights(j, i) * input(i);
        if(with_bias) {
            output(i) += bias(i);
        }
    }
    void print_linear_dim() {
        std::cout << "Linear: (" << in_size << ", " << out_size << ")" << std::endl;
    }
    void linear_3d(taco::Tensor<T> &output, const taco::Tensor<T> &input) {
        #ifdef DEBUGLINEAR
        for(auto dim : input.getDimensions()) {
            std::cout << dim << std::endl;
        }
        std::cout << std::endl << std::endl; 
        std::cout << "weights: " << std::endl;
        for(auto dim : weights_transposed.getDimensions()) {
            std::cout << dim << std::endl;
        }
        #endif
        // shape(weights_transposed, "WT");
        // output(i, j, l) = input(i, j, k) * weights(k, l);
        output(i, j, l) =  weights(k, l) * input(i, j, k);

        if(with_bias) {
            // output(i, j, k) += bias(k);
        }
    }
    void linear_4d(taco::Tensor<T> &output, const taco::Tensor<T> &input) {
        // shape(input, input.getName());
        // shape(weights, weights.getName());

        output(i, j, k, m) = weights(l, m) * input(i, j, k, l);
        if(with_bias) {
            output(i, j, k, l) += bias(l);
        }
    }
    void forward(taco::Tensor<T> &output, const taco::Tensor<T> &input) {
        // Dot product of input and weights transposed
        int num_dims = input.getOrder();
        if(num_dims == 1) {
            linear_1d(output, input);
        }
        else if(num_dims == 3) {
            linear_3d(output, input);
        }
        else if(num_dims == 4) {
            linear_4d(output, input);
        }
        // output(i) = input(i) * weights(j, i);

        // // Add bias if any, likely won't need it
        // if(with_bias) {
        //     output(i) += bias(i);
        // }
    }

    taco::Tensor<T> forward(const taco::Tensor<T> &input) {
        if(input.getOrder() == 3) {
            taco::Tensor<T> output({input.getDimension(0), weights.getDimension(0), weights.getDimension(1)}, dense);
            forward(output, input);
            return output;
        }
        else {
            taco::Tensor<T> output({input.getDimensions()}, dense);
            forward(output, input);
            return output;
        }
    }

    void set_weights(float val) {
        for (int i = 0; i <in_size; ++i) { 
            for(int j = 0; j < out_size; ++j) {
                // weights.insert({i, j}, (T)(dist(seed) * std::sqrt(2.0f / in_size)));
                // weights.insert({i, j}, (T)dist(seed));
                weights.insert({i, j}, (T)(val));
            }
        }
    }

    void initialize_weights() {
        // TODO: Change back to time(0) based seed
        std::default_random_engine seed(time(NULL));
        // std::normal_distribution<T> dist(0, 1);
        T stdv = 1.0f / std::sqrt((T)in_size);
        std::uniform_real_distribution<T> dist(-stdv, stdv);

        // Initialize weights using normal distribution with uniform(-1/sqrt(in_features), 1/sqrt(in_features))
        // multiply sample with sqrt of (2 / n) n -> input size 
        for (int i = 0; i <in_size; ++i) { 
            for(int j = 0; j < out_size; ++j) {
                weights.insert({i, j}, (T)(dist(seed) * std::sqrt(2.0f / in_size)));
                //weights.insert({i, j}, (T)(0.2));
            }
        }
        weights.pack();
        // if(with_bias) {
        //     taco::Tensor<T> temp_bias({out_size}, dense);
        //     bias = temp_bias;
        //     for(int i = 0; i < out_size; ++i) {
        //         bias.insert({i}, (T)dist(seed));
        //     }
        //     bias.pack();
        // }

        // #ifdef DEBUG
        // std::cout << weights << std::endl;
        // #endif
        weights_initialized = true;
    }

    taco::Tensor<T> get_weights() { return weights; }

private:
    taco::Tensor<T> weights;
    taco::Tensor<T> weights_transposed;
    taco::Tensor<T> bias;
    int in_size;
    int out_size;
    bool with_bias;
    bool weights_initialized; // might not be needed, could always generate instead of passing in
    bool bias_initialized; // Same as above
    bool transposed;
};

#endif //TRANSFORMER_LINEAR_H
