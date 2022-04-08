#ifndef TRANSFORMER_DROPOUT_H
#define TRANSFORMER_DROPOUT_H

#include "common.h"
#include "layer.h"

// using namespace taco;

template<typename T>
class Dropout : public Layer<T> {
public:
    // TODO: Maybe make input and output shapes global so I don't have to pass it in
    Dropout(float drop_prob, std::string name = "dropout") : Layer<T>(name), drop_prob(drop_prob), initialized(false)
    {
        // TODO: Initialize in constructor
    }

    void generate_inv_prob(int row_dim, int col_dim) {
        taco::Tensor<T> dropout_inv("dropout_mat", {row_dim, col_dim}, {dense, dense});
        std::default_random_engine seed(time(0));
        std::uniform_real_distribution<T> dist(0, 1);
        float keep_prob = 1.0 - drop_prob;

        // TODO: Figure out syntactic sugar for initializing 
        for(int i = 0; i < row_dim; ++i) {
            for(int j = 0; j < col_dim; ++j) { 
                dropout_inv.insert({i, j}, (T)(dist(seed) < keep_prob));
            }
        }
        drop = dropout_inv;
        initialized = true;
    }

    void generate_inv_prob_vec(int row_dim) { 
        taco::Tensor<T> dropout_inv("dropout_vec", {row_dim}, dense);
        std::default_random_engine seed(120);
        std::uniform_real_distribution<T> dist(0, 1);
        float keep_prob = 1.0 - drop_prob;
        for(int i = 0; i < row_dim; ++i) {
            dropout_inv.insert({i}, (T)(dist(seed) < keep_prob));
        }
        drop = dropout_inv;
        initialized = true;
    }

    void forward(taco::Tensor<T> &output, const taco::Tensor<T> &input) 
    {
        // Element-wise multiplication with drop where each entry is 0 with probability drop_prob
        // and 1 with probability (1 - drop_prob)
        int dim = input.getOrder();
        if(!initialized) {
            if(dim == 2) {
                generate_inv_prob(input.getDimension(0), input.getDimension(1));
            }
            else if(dim == 3) {
                generate_inv_prob(input.getDimension(1), input.getDimension(2));
            }
        }
        float scale = 1 / (1 - drop_prob);
        if(dim == 2) {
            output(i, j) = input(i, j) * drop(i, j) * scale;
        }
        if(dim == 3) {
            output(i, j, k) = input(i, j, k) * drop(j, k) * scale;
        }
    }

    // void forward(taco::Tensor<T> &input)
    // {
    //     // Element-wise multiplication with drop where each entry is 0 with probability drop_prob
    //     // and 1 with probability (1 - drop_prob)

    //     int dim = input.getOrder();
    //     if(!initialized) {
    //         if(dim == 1) {
    //             generate_inv_prob_vec(input.getDimension(0));
    //         }
    //         else {
    //             generate_inv_prob(input.getDimension(0), input.getDimension(1));
    //         }
    //     }
    //     float scale = 1 / (1 - drop_prob);
    //     if(dim == 1) {
    //         taco::Tensor<T> temp("temp", {input.getDimension(0)}, dense);
    //         temp(i) = input(i);
    //         input(i) = temp(i) * drop(i) * scale;
    //     }
    //     else if (dim == 2) {
    //         taco::Tensor<T> temp("temp", input.getDimensions(), {dense, dense});
    //         temp(i, j) = input(i, j);
    //         input(i, j) = temp(i, j) * drop(i, j) * scale;
    //     }
    //     else if (dim == 3) {

    //     }
    // }

    taco::Tensor<T> forward(const taco::Tensor<T> &input) 
    {
        // Element-wise multiplication with drop where each entry is 0 with probability drop_prob
        // and 1 with probability (1 - drop_prob)
        taco::Tensor<T> output(input.getDimensions(), dense);
        forward(output, input);
        return output;
    }

private:
    taco::Tensor<T> drop; 
    float drop_prob;
    bool initialized;
};

#endif //TRANSFORMER_DROPOUT_H
