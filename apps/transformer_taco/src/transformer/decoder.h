#ifndef TRANSFORMER_DECODER_H
#define TRANSFORMER_DECODER_H

#include "common.h"

// using namespace taco;

template<typename T>
class PositionEncoding : public Layer<T> {
public:
    PositionEncoding() {}
    void forward(taco::Tensor<T> &output, taco::Tensor<T> &input) {
        // TODO: NOT IMPLEMENTED YET
    }
};

#endif //TRANSFORMER_DECODER_H