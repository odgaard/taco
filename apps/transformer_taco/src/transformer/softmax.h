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
    void forward(taco::Tensor<T> &output, const taco::Tensor<T> &input) {
        taco::Tensor<T> exp_sum("exp_sum", {input.getDimensions()[0]}, dense);
        taco::Tensor<T> result("result");

        // TODO: Numerically unstable version, need max reduction for stable
        exp_sum(i) = exp(input(i, j) - maxElem(input, 1));
        output(i,j) = exp(input(i, j)) / exp_sum(i);
    }
    void forward(taco::Tensor<T> &output, const taco::Tensor<T> &input, const int dim) {
        taco::Tensor<T> exp_sum("exp_sum", {input.getDimension(0), input.getDimension(1), input.getDimension(2)}, dense);
        taco::Tensor<T> temp("temp", {input.getDimensions()}, dense);
        taco::Tensor<T> max_elem("max_elem", {input.getDimension(0), input.getDimension(1), input.getDimension(2)}, dense);

        // FIXME: Need to subtract max along axis specified by dim first
        // max_elem = max(input, dim);
        max(max_elem, input, dim);
        max_elem.evaluate();
        // max_elem(i, j, k) = sum(l, input(i, j, k, l)) / (T)(input.getDimension(2));

        #ifdef DEBUG
        std::cout << max_elem << std::endl;
        std::cout << "max in softmax";
        std::cin.ignore();
        #endif
        
        temp(i, j, k, l) = exp(input(i, j, k, l) - max_elem(i, j, k));

        #ifdef DEBUG
        taco::Tensor<T> slicer({1, 1, 1, 64}, dense);
        taco::Tensor<T> input_copy = input;
        slicer(i, j, k, l) = input_copy(i(0, 1), j(0, 1), k(0, 1), l);
        
        std::cout << slicer << std::endl;
        std::cout << "output";
        std::cin.ignore();
        #endif

        exp_sum(i, j, k) = sum(l, temp(i, j, k, l));
        exp_sum.evaluate(); 
        // std::cout << exp_sum << std::endl;
        output(i, j, k, l) = temp(i, j, k, l) / exp_sum(i, j, k);
        // std::cout << output << std::endl;
    }
};

#endif //TRANSFORMER_SOFTMAX_H