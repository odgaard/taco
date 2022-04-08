#ifndef TRANSFORMER_ATTENTION_H
#define TRANSFORMER_ATTENTION_H

#include <math.h>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xbuffer_adaptor.hpp>

#include "common.h"
#include "softmax.h"
#include "layer.h"
#include "linear.h"

template<typename T>
class MultiHeadedAttention {
public:
    MultiHeadedAttention(int embed_dim, int heads,
                         std::string name = "Self-attention")
        : embed_dim(embed_dim), heads(heads),
          heads_dim(embed_dim / heads), values(heads_dim, heads_dim),
          keys(heads_dim, heads_dim), queries(heads_dim, heads_dim),
          fc_out(heads * heads_dim, embed_dim, false) {
        if (heads_dim * heads != embed_dim) {
            std::cout << "Incompatible dimensions for number of heads" << std::endl;
            exit(1);
        }
    }

    void shape(taco::Tensor<T> &tensor, std::string name) {
        std::cout << name << std::endl;
        for(auto dim : tensor.getDimensions()) {
            std::cout << dim << std::endl;
        }
        std::cout << std::endl;
    }

    void forward(taco::Tensor<T> &attention, const taco::Tensor<T> &Q, const taco::Tensor<T> &K, const taco::Tensor<T> &V) {
        int batch_size = Q.getDimension(0);
        int max_seq = Q.getDimension(1);
        taco::Tensor<T> multi_Q = split_heads(Q, {batch_size, max_seq, heads, heads_dim});
        taco::Tensor<T> multi_K = split_heads(K, {batch_size, max_seq, heads, heads_dim});
        taco::Tensor<T> multi_V = split_heads(V, {batch_size, max_seq, heads, heads_dim});

#ifdef DEBUG1
        std::cout << multi_Q << std::endl;
        std::cout << "After multi";
        std::cin.ignore();
#endif

        taco::Tensor<T> query = queries.forward(multi_Q);
        taco::Tensor<T> key = keys.forward(multi_K);
        taco::Tensor<T> value = values.forward(multi_V);

#ifdef DEBUG1
        std::cout << query << std::endl;
        std::cout << "query After linear";
        std::cin.ignore();

        std::cout << key << std::endl;
        std::cout << "key After linear";
        std::cin.ignore();

        std::cout << value << std::endl;
        std::cout << "value After linear";
        std::cin.ignore();
#endif

        // taco::Tensor<T> multihead = scaled_dot_product_attention(multi_Q, multi_K, multi_V);
        // attention = join_heads(multihead);

        taco::Tensor<T> multihead = scaled_dot_product_attention(query, key, value);
        taco::Tensor<T> temp_attention = join_heads(multihead);

#ifdef DEBUG
        std::cout << temp_attention << std::endl;
        std::cout << "After scaled dot prod att";
        std::cin.ignore();
#endif

        fc_out.forward(attention, temp_attention);

        // std::cout << attention << std::endl;
        // std::cout << "After fc out";
        
        // std::cin.ignore();

#ifdef DEBUG
        std::cout << attention << std::endl;
        std::cout << "After fc out";
        
        std::cin.ignore();
#endif
        // std::cout << attention << std::endl;
        // exit(1);
    }

    taco::Tensor<T> scaled_dot_product_attention(const taco::Tensor<T> &Q, const taco::Tensor<T> &K, const taco::Tensor<T>& V) {
        int batch_size = Q.getDimension(0);
        int query_len = Q.getDimension(1);
        int heads = Q.getDimension(2);
        int heads_dim = Q.getDimension(3);
        int key_len = K.getDimension(1);
        // int embed_size = heads_dim * heads;

        T scale = (T)(1.0 / std::sqrt(heads_dim * heads));
        // i: batch, j: heads, k: query_len, l: key_len, m: heads_dim
        taco::Tensor<T> QK({batch_size, heads, query_len, key_len}, dense); 
        taco::Tensor<T> temp_QK({batch_size, heads, query_len, key_len}, dense); 

        temp_QK(i, j, k, l) = (Q(i, k, j, m) * K(i, l, j, m)); 

        taco::Tensor<T> softmax_QK({batch_size, heads, query_len, key_len}, dense);
        Softmax<T> softmax;
        QK(i, j, k, l) = temp_QK(i, j, k, l) * scale;
        softmax.forward(softmax_QK, QK, 3);

        // std::cout << Q << std::endl;
        // std::cout << "Q" << std::endl;
        // std::cin.ignore();
        
        // std::cout << K << std::endl;
        // std::cout << "K" << std::endl;
        // std::cin.ignore();
        
        // std::cout << V << std::endl;
        // std::cout << "V" << std::endl;
        // std::cin.ignore();
        
        #ifdef DEBUG

        std::cout << QK << std::endl;
        std::cout << "QK" << std::endl;
        std::cin.ignore();    

        std::cout << softmax_QK << std::endl;
        std::cout << "Softmax result" << std::endl;
        std::cin.ignore();

        taco::Tensor<T> slicer({1, 1, 1, 64}, dense);
        taco::Tensor<T> input_copy = temp_QK;
        slicer(i, j, k, l) = input_copy(i(0, 1), j(0, 1), k(0, 1), l);
        
        std::cout << slicer << std::endl;
        std::cout << "temp_QK slice";
        std::cin.ignore();

        std::cout << K << std::endl;
        std::cout << "K";
        std::cin.ignore();

        taco::Tensor<T> slicer2({1, 1, 1, 16}, dense);
        taco::Tensor<T> input_copy2 = K;
        slicer2(i, j, k, l) = input_copy2(i(0, 1), j(0, 1), k(0, 1), l);
        
        std::cout << slicer2 << std::endl;
        std::cout << "K slice";
        std::cin.ignore();

        #endif

        taco::Tensor<T> multihead({batch_size, query_len, heads, heads_dim}, dense);
        multihead(i, k, j, m) = softmax_QK(i, j, k, l) * V(i, l, j, m);
        // Concat heads dimension along heads dim

        return multihead;
    }

    taco::Tensor<T> split_heads(const taco::Tensor<T> &input, const std::vector<int> &dims) {
        taco::Tensor<T> output = reshape(input, dims);
        // return output.transpose({0, 2, 1, 3});
        return output;
    }

    taco::Tensor<T> join_heads(const taco::Tensor<T> &input) {
        // taco::Tensor<T> output = input.transpose({0, 2, 1, 3});
        taco::Tensor<T> temp_output = input;
        // shape(temp_output, "input");
        // std::cout << std::endl;
        // for(auto dim : input.getDimensions())
        //     std::cout << dim << std::endl;
        // std::cout << std::endl;
        taco::Tensor<T> output =  reshape(input, {temp_output.getDimension(0), temp_output.getDimension(1), 
                                          temp_output.getDimension(2) * temp_output.getDimension(3)});
        // std::cout << output << std::endl;
        // std::cout << "After out";
        // std::cin.ignore();

        // taco::Tensor<T> slicer2({1, 1, 64}, dense);
        // taco::Tensor<T> input_copy2 = output;
        // slicer2(i, j, k) = input_copy2(i(0, 1), j(0, 1), k);
        // std::cout << slicer2 << std::endl;
        // std::cout << "Out slice";
        // std::cin.ignore();

        return output;
    }

private:
    int batch_size;
    int max_seq;
    int heads;
    int embed_dim;
    int heads_dim;
    Linear<T> values;
    Linear<T> keys;
    Linear<T> queries;
    Linear<T> fc_out;
};


#endif //TRANSFORMER_ATTENTION_H
