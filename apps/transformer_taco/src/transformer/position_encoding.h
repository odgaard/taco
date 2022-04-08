#ifndef TRANSFORMER_POSITIONENCODING_H
#define TRANSFORMER_POSITIONENCODING_H

#include "common.h"
#include "layer.h"
#include "dropout.h"

// using namespace taco;

template<typename T>
class PositionEncoding : public Layer<T> {
public:
    PositionEncoding(int max_len, int embed_size, std::string name = "Position Encoding") : Layer<T>(name), 
                                                                                            posEncoding("positionEncoding", {max_len, embed_size}, dense)
    {
        taco::Tensor<T> position("test", {max_len}, dense);
        taco::Tensor<T> div_term("div_term", {embed_size / 2}, dense);
        taco::Tensor<T> temp("temp", {embed_size / 2}, dense);

        arange(position, 0, max_len, 1);
        arange(temp, 0, embed_size / 2, 2);

        div_term(k) = exp(temp(k) * (-std::log(10000.0)) / embed_size);

        posEncoding(i, j(0, embed_size, 2)) = sin(position(i) * div_term(j));
        posEncoding.evaluate();
        posEncoding(i, j(1, embed_size, 2)) = cos(position(i) * div_term(j));
        posEncoding.compile();
        posEncoding.compute();

    }
    void forward(taco::Tensor<T> &output, const taco::Tensor<T> &input) {
        // Add position encoding to input embedding
        output(i, j) = input(i, j) + posEncoding(i, j(0, input.getDimensions()[0]));
    }
    taco::Tensor<T> forward(const taco::Tensor<T> &input) {
        // Add position encoding to input embedding
        taco::Tensor<T> output("output", input.getDimensions(), dense);
        #ifdef DEBUG
        std::cout << posEncoding << std::endl;
        std::cout << "PE";
        std::cin.ignore();
        #endif
        output(i, j, k) = input(i, j, k) + posEncoding(j, k(0, input.getDimensions()[2]));
        return output;
    }
private:
    // Dropout<T> drop;
    taco::Tensor<T> posEncoding;
};

#endif //TRANSFORMER_POSITIONENCODING_H