#ifndef TRANSFORMER_LAYER_H
#define TRANSFORMER_LAYER_H

#include <array>
#include "common.h"

// using namespace taco;

template<typename T>
class Layer {
public:
    Layer() {}
    Layer(const std::string& name) : _name(name) {}
    virtual void forward(taco::Tensor<T> &output, const taco::Tensor<T> &input) = 0;

    friend std::ostream& operator<<(std::ostream &out, const Layer &layer) {
        return out << "Layer: " << layer.get_name() << std::endl;
    }

    virtual std::string get_name() { return _name; }

private:
    // TODO: Add metadata info for verbose printing of Layer
    std::string _name;
    // int _num_params;
};

#endif //TRANSFORMER_LINEAR_H