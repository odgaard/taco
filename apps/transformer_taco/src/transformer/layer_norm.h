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
    void forward(taco::Tensor<T> &output, const taco::Tensor<T> &input) {
        // Changed to batched 
        int first_dim = input.getDimension(0);
        int second_dim = input.getDimension(1);
        int third_dim = input.getDimension(2);
        taco::Tensor<T> mean("mean", {first_dim, second_dim}, dense);
        taco::Tensor<T> var("var", {first_dim, second_dim}, dense);

        // Calculate mean over both matrix dimension
        // mean(i) = sum(j, sum(k, input(i, j, k))) / (T)(second_dim * third_dim);
        mean(i, j) = sum(k, input(i, j, k)) / (T)(third_dim);
        mean.evaluate();

        // Var(x) = E(x^2) - E(x)^2
        // var(i) = sum(j, sum(k, pow(input(i, j, k) - mean(i), (T)2.0))) / (T)(second_dim * third_dim);
        var(i, j) = sum(k, pow(input(i, j, k) - mean(i, j), (T)2.0)) / (T)(third_dim);
        var.evaluate();
        
        // output = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
        // taco::Tensor<T> temp_var(var.getDimensions(), var.getFormat());
        // taco::Tensor<T> sqrt_var(var.getDimensions(), var.getFormat());

        // temp_var(i, j) = eps + var(i, j);
        // sqrt_var(i, j) = sqrt(temp_var(i, j));

        // output(i, j, k) = ((input(i, j, k) - mean(i, j)) / sqrt_var(i, j)) * gamma + beta;
        output(i, j, k) = ((input(i, j, k) - mean(i, j)) / sqrt(eps + var(i, j))) * gamma + beta;

        // std::cout << input << std::endl;
        // std::cout << "Layer norm in";
        // std::cin.ignore();

        // output(i, j, k) = (input(i, j, k) - mean(i)) / (sqrt(var(i) + eps) * gamma + beta);
    }
    taco::Tensor<T> forward(const taco::Tensor<T> &input) {
        taco::Tensor<T> output(input.getDimensions(), dense);

        forward(output, input);
        
        return output;
    }

private:
    T eps;
    T gamma;
    T beta;
    int size;
};

#endif //TRANSFORMER_LAYERNORM_H