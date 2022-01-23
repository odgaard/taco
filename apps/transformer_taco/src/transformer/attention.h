#ifndef TRANSFORMER_ATTENTION_H
#define TRANSFORMER_ATTENTION_H

#include "common.h"
#include <math.h>
#include <vector>

template<typename T>
static taco::Tensor<T> split_heads(taco::Tensor<T> &input, const std::vector<int> &dims) {
    //TODO: Will not work as no reshape in TACO
    taco::Tensor<T> output = reshape(input, dims);
    return output.transpose({0, 2, 1, 3});
}

template<typename T>
static taco::Tensor<T> join_heads(taco::Tensor<T> &input) {
    // taco::Tensor<T> output = transpose(input, {0, 2, 1, 3});
    taco::Tensor<T> output = input.transpose({0, 2, 1, 3});
    //TODO: Will not work as no reshape in TACO
    return reshape(output, output.getDimensions());
}



template<typename T>
taco::Tensor<T> scaled_dot_product_attention(const taco::Tensor<T> &Q, const taco::Tensor<T> &K, const taco::Tensor<T>& V, int n_key) {
    T scale = (T)(1.0 / std::sqrt(n_key));
    taco::Tensor<T> scaled_Q(Q);
    scaled_Q(i, j, k) = scaled_Q(i, j, k) * scale;
    taco::Tensor<T> QK(Q); // Get proper shape
    // TODO: Figure out how to split heads or modify algorithm
    // NVIDIA Apex claims to have a method for splitting without reshapes

    // i: batch, j: heads, k: query_len, l: key_len, m: heads_dim
    // TODO: Below expression might not be accurate, needs revising
    QK(i, j, k, l) = scaled_Q(i, k, j, m) * K(i, j, k, l); // TODO: Figure out proper shapes/necessary ops

    taco::Tensor<T> softmax_QK(QK);
    Softmax<T> softmax_layer();
    softmax_layer.forward(softmax_QK, QK);

    // TODO: Maybe use dropout before
    taco::Tensor<T> multihead = softmax_QK(i, j) * V(j, k);

    return multihead;
}

template<typename T>
taco::Tensor<T> attention(taco::Tensor<T> &Q, taco::Tensor<T> &K, taco::Tensor<T>& V,
                          int batch_size, int max_seq, int n_heads, int n_model, int n_key) {
    taco::Tensor<T> multi_Q = split_heads(Q, {batch_size, max_seq, n_heads, n_model/n_heads});
    taco::Tensor<T> multi_K = split_heads(K, {batch_size, max_seq, n_heads, n_model/n_heads});
    taco::Tensor<T> multi_V = split_heads(V, {batch_size, max_seq, n_heads, n_model/n_heads});

    taco::Tensor<T> multihead = scaled_dot_product_attention(multi_Q, multi_K, multi_V, n_key);
    return join_heads(multihead);
}

#endif //TRANSFORMER_ATTENTION_H