#include <iostream>
#include <algorithm>
#include <memory>

#include "attention.h"
#include "common.h"

int main(int argc, char* argv[]) {
    int batch_size = 32;
    int max_seq = 512;
    int n_heads = 8;
    int n_key = 64;
    int n_model = n_heads * n_key;

    taco::Tensor<float> Q("Q", {batch_size, max_seq, n_model}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense, taco::ModeFormat::Dense});
    taco::Tensor<float> K("K", {batch_size, max_seq, n_model}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense, taco::ModeFormat::Dense});
    taco::Tensor<float> V("V", {batch_size, max_seq, n_model}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense, taco::ModeFormat::Dense});

    auto joined_heads = attention(Q, K, V, batch_size, max_seq, n_heads, n_model, n_key);

    return 0;
}