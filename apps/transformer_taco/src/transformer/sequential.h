#ifndef TRANSFORMER_SEQUENTIAL_H
#define TRANSFORMER_SEQUENTIAL_H

#include <array>
#include "common.h"

// using namespace taco;

template<typename T>
class Sequential {
public:
    Sequential(int num_layers, int hidden_size) : _num_layers(num_layers), _layers(num_layers), _hidden_size(hidden_size) {

    }

    void add_layer(const Layer<T> &layer) {
        _layers.push_back(layer);
    }
    
    void operator()(taco::Tensor<T> &output, const taco::Tensor<T> &input) {
        if(_num_layers == 0) {
            std::cout << "No layers in sequential block" << std::endl;
            exit(0);
        }
        // TODO: Use vector of tensors instead
        // taco::Tensor<T> tmp_out("tmp_out", {_num_layers - 1, _hidden_size}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense})
        std::vector<taco::Tensor<T>> tmp_out;
        for(int i = 0; i < _num_layers; ++i) {
            taco::Tensor<T> tmp("tmp" + std::to_string(i), {_hidden_size}, taco::Format{taco::ModeFormat::Dense});
            tmp_out.push_back(tmp);
        }
        for (int i = 0; i < _num_layers; ++i) {
            if(i == 0) {
                _layers[i].forward(tmp_out[i], input);
            }
            else if(i == _num_layers - 1) {
                _layers[i].forward(output, tmp_out[i - 1]);
            }
            else {
                _layers[i].forward(tmp_out[i], tmp_out[i - 1]);
            }
        }
    }

    friend std::ostream& operator<<(std::ostream &out, const Sequential &sequential) {
        return out << "Layers: " << sequential.get_layers() << std::endl;
    }

    Layer<T> get_layer(int layer_idx) { return _layers[layer_idx]; }

    std::vector<Layer<T>> get_layers() { return _layers; }

private:
    // TODO: Add metadata info for verbose printing of Layer
    std::vector<Layer<T>> _layers;
    int _num_layers;
    int _hidden_size;
    // int _num_params;
};

#endif //TRANSFORMER_SEQUENTIAL_H