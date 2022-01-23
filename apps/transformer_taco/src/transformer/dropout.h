#ifndef TRANSFORMER_DROPOUT_H
#define TRANSFORMER_DROPOUT_H

#include "common.h"

// using namespace taco;

template<typename T>
class Dropout : public Layer<T> {
public:
    // TODO: Maybe make input and output shapes global so I don't have to pass it in
    Dropout(float drop_prob, int x, int y, std::string name = "dropout") : Layer<T>(name), drop_prob(drop_prob), dim_x(x), dim_y(y), 
                                                                           drop("drop", {x, y}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense}) 
                                                                           //TODO: Experiment with different formats
    {
        std::default_random_engine seed(time(0));
        std::normal_distribution<T> dist(0, 1);
        float keep_prob = 1 - drop_prob;
        for(int i = 0; i < dim_x; ++i) {
            for(int j = 0; j < dim_y; ++j) { 
                drop.insert({i, j}, (T)(dist(seed) < keep_prob));
            }
        }
    }
    void forward(taco::Tensor<T> &output, taco::Tensor<T> &input) 
    {
        // Element-wise multiplication with drop where each entry is 0 with probability drop_prob
        // and 1 with probability (1 - drop_prob)
        output(i, j) = input(i, j) * drop(i, j);
    }

private:
    taco::Tensor<T> drop; 
    float drop_prob;
    int dim_x;
    int dim_y;
};

#endif //TRANSFORMER_DROPOUT_H