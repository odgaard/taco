#ifndef TRANSFORMER_LAYERNORM_H
#define TRANSFORMER_LAYERNORM_H

#include <array>
#include "common.h"
#include "layer.h"

// using namespace taco;

template<typename T>
class LayerNorm : public Layer<T>{
public:
    // Using Pytorch default elementwise_affine values (gamma/weights = 1.0, beta/bias = 0.0)
    LayerNorm(T eps = 1e-5, T gamma = 1.0, T beta = 0.0, std::string name = "Layer Norm") : Layer<T>(name), eps(eps), gamma(gamma), beta(beta){}
    void forward(taco::Tensor<T> &output, taco::Tensor<T> &input) {
        int row_dim = input.getDimensions()[0];
        int col_dim = input.getDimensions()[1];
        taco::Tensor<T> mean("mean");
        taco::Tensor<T> var("var");

        // Calculate mean over both matrix dimension
        mean = sum(i, sum(j, input(i, j))) / (T)(row_dim * col_dim);
        mean.evaluate();

        // Var(x) = E(x^2) - E(x)^2
        var = sum(i, sum(j, pow(input(i, j) - mean(), 2.0f))) / (T)(row_dim * col_dim);
        var.evaluate();

        // output = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
        output(i, j) = (input(i, j) - mean()) / (sqrt(var() + eps) * gamma + beta);
    }

private:
    // TODO: Add metadata info for verbose printing of Layer
    std::string _name;
    T eps;
    T gamma;
    T beta;
    int size;
};

#endif //TRANSFORMER_LAYERNORM_H