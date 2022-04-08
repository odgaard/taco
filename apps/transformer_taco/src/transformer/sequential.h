#ifndef TRANSFORMER_SEQUENTIAL_H
#define TRANSFORMER_SEQUENTIAL_H

#include <array>
#include "common.h"
#include "layer.h"

// using namespace taco;

template<typename T>
class Sequential {
public:
    Sequential(int hidden_size) : _num_layers(0), _hidden_size(hidden_size) {

    }

    void add_layer(Layer<T> *layer) {
        _layers.push_back(layer);
        _num_layers++;
    }
    
    void operator()(taco::Tensor<T> &output, taco::Tensor<T> &input) {
        if(_num_layers == 0) {
            std::cout << "No layers in sequential block" << std::endl;
            exit(0);
        }
        // TODO: Use vector of tensors instead
        // taco::Tensor<T> tmp_out("tmp_out", {_num_layers - 1, _hidden_size}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense})
        std::vector<taco::Tensor<T>> tmp_out;
        
        // for(int i = 0; i < _num_layers; ++i) {
            taco::Tensor<T> tmp("tmp", {_hidden_size}, dense);
            taco::Tensor<T> tmp2("tmp2", {_hidden_size}, dense);
            // tmp_out.push_back(tmp);
        // }
        // TODO: Reuse temp by copying input to temp
        for (int i = 0; i < _num_layers; ++i) {
            if(i == 0 && _num_layers != 1) {
                // std::cout << *_layers[i] << std::endl;
                _layers[i]->forward(tmp, input);
                // _layers[i]->forward(tmp_out[i], input);
            }
            else if (_num_layers == 1) {
                std::cout << "inside layer\n";
                _layers[i]->forward(output, input);
            }
            else if(i == _num_layers - 1) {
                // std::cout << _layers[i] << std::endl;
                _layers[i]->forward(output, tmp2);
                // _layers[i]->forward(output, tmp_out[i - 1]);
            }
            else {
                // std::cout << _layers[i] << std::endl;
                _layers[i]->forward(tmp2, tmp);
                tmp(j) = tmp2(j);
                // _layers[i]->forward(tmp_out[i], tmp_out[i - 1]);
            }
        }

        // printTensor(tmp_out[0]);
        // printTensor(output);
        std::cout << input << std::endl;
        std::cout << output << std::endl;
    }

    friend std::ostream& operator<<(std::ostream &out, const Sequential &sequential) {
        return out << "Layers: " << sequential.get_layers() << std::endl;
    }

    // Layer<T> get_layer(int layer_idx) { return _layers[layer_idx]; }

    std::vector<Layer<T>> get_layers() { return _layers; }

private:
    // TODO: Add metadata info for verbose printing of Layer
    std::vector<Layer<T>*> _layers;
    int _num_layers;
    int _hidden_size;
    // int _num_params;
};

#endif //TRANSFORMER_SEQUENTIAL_H