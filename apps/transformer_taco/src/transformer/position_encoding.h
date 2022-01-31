#ifndef TRANSFORMER_POSITIONENCODING_H
#define TRANSFORMER_POSITIONENCODING_H

#include "common.h"
#include "layer.h"
#include "dropout.h"

// using namespace taco;

// TODO: Change last param to colum vector bool
template<typename T>
static void arange(taco::Tensor<T> &in, int low, int high, int step = 1, int num_indices=1) {
    for(int i = low; i < high; i += step) {
        if(num_indices == 1) {
            in.insert({i}, (T)i);
        }
        else {
            in.insert({i, 0}, (T)i);
        }
    } 
}

template<typename T>
class PositionEncoding : public Layer<T> {
public:
    PositionEncoding(int emb_size, float dropout, int max_len = 5000, std::string name = "Position Encoding") : Layer<T>(name), 
                                                                                                               drop(dropout, max_len, emb_size),
                                                                                                               positionEncoding("positionEncoding", {max_len, emb_size}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense})
    {
        taco::Tensor<T> position("position", {max_len, 1}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense});
        taco::Tensor<T> div_term("div_term", {emb_size}, taco::Format{taco::ModeFormat::Dense});
        taco::Tensor<T> temp("temp", {emb_size}, taco::Format{taco::ModeFormat::Dense});

        arange(position, 0, max_len, 1, 2);
        arange(div_term, 0, emb_size, 2);

        std::cout << position << std::endl;

        temp(i) = exp(div_term(i) * (-std::log(10000.0)) / emb_size);

        // TODO: Figure out division term for (position / (10000 ^(2i / emb_size)))
        // TODO: Might need to use array algebra and infer indices for each position if possible
        positionEncoding(i, j(0, max_len, 2)) = sin(position(i, k) * temp(i));
        positionEncoding(i, j(1, max_len, 2)) = cos(position(i, k) * temp(i));

    }
    void forward(taco::Tensor<T> &output, taco::Tensor<T> &input) {
        // Add position encoding to input embedding
        std::cout << positionEncoding << std::endl;
        output(i, j) = input(i, j) + positionEncoding(i, j(0, input.getDimensions()[0]));
        // Apply dropout, typically done after this step
        // TODO: Might be better to have another forward function with one input that is overriden
        // drop.forward(output, output);
    }
private:
    Dropout<T> drop;
    taco::Tensor<T> positionEncoding;
};

#endif //TRANSFORMER_POSITIONENCODING_H