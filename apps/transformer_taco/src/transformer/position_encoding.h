#ifndef TRANSFORMER_POSITIONENCODING_H
#define TRANSFORMER_POSITIONENCODING_H

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

#endif //TRANSFORMER_POSITIONENCODING_H