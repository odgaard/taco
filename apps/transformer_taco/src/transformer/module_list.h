#ifndef TRANSFORMER_MODULE_LIST_H
#define TRANSFORMER_MODULE_LIST_H

#include "common.h"
#include "layer.h"

// using namespace taco;

template<typename moduleType, typename T>
class ModuleList {
public:
    ModuleList(int size) : m_length(size)
    {
        if(size <= 0) {
            std::cout << "Invalid module size" << std::endl;
        }
    }
    void push(moduleType &module) { 
        modules.push_back(module);
    }
    taco::Tensor<T> operator()(const taco::Tensor<T> &x, const taco::Tensor<T> &value, const taco::Tensor<T> &key, 
                                const taco::Tensor<T> &src_mask, const taco::Tensor<T> &trg_mask) {
        // taco::Tensor<T> temp_x(x.getDimensions(), dense);
        taco::Tensor<T> temp_x = x;
        int count = 0;
        for(auto module : modules) {
            #ifdef DEBUG
            std::cout << temp_x << std::endl;
            std::cout << "Press Enter to Continue";
            std::cin.ignore();
            #endif
            taco::Tensor<T> temp = module(temp_x, value, key, src_mask, trg_mask);    
            temp_x = temp;
        }
        return temp_x;
    }

private:
    int m_length; // length
    std::vector<moduleType> modules;
};

#endif //TRANSFORMER_MODULE_LIST_H