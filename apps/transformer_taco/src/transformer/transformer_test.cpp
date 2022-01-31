#include <iostream>
#include <algorithm>
#include <memory>

#include "attention.h"
#include "common.h"
#include "relu.h"
#include "dropout.h"
#include "layer_norm.h"
#include "position_encoding.h"

void test_relu() { 
    taco::Tensor<float> A("A", {2, 2}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense});
    taco::Tensor<float> B("B", {2, 2}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense});
    std::vector<float> c{20.0, -1.0, -1.0, -1.0};
    for(int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            A.insert({i, j}, (float)c[i * 2 + j]);
        }
    }
    A.pack();

    ReLU<float> test_;
    test_.forward(B, A);

    std::cout << A << std::endl;
    std::cout << B << std::endl;
}

void test_position() { 
    taco::Tensor<float> A("A", {4, 4}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense});
    taco::Tensor<float> B("B", {4, 4}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense});
    std::vector<float> c{20.0, -1.0, -1.0, -1.0, 1.0, 1.0, 10.0, 10.0, 5.0, 5.0, 5.0, .5, .5, .5, .5, .5};
    for(int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            A.insert({i, j}, (float)c[i * 4 + j]);
        }
    }
    A.pack();

    std::cout << "Entering pE" << std::endl;

    PositionEncoding<float> test_(4, 0.5, 4);
    test_.forward(B, A);

    std::cout << A << std::endl;
    std::cout << B << std::endl;
}

void test_softmax() { 
    taco::Tensor<float> A("A", {2, 2}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense});
    taco::Tensor<float> B("B", {2, 2}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense});
    std::vector<float> c{20.0, -1.0, -1.0, -1.0};
    for(int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            A.insert({i, j}, (float)c[i * 2 + j]);
        }
    }
    A.pack();

    Softmax<float> test_;
    test_.forward(B, A);

    std::cout << A << std::endl;
    std::cout << B << std::endl;
}

void test_dropout() { 
    taco::Tensor<float> A("A", {2, 2}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense});
    taco::Tensor<float> B("B", {2, 2}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense});
    std::vector<float> c{20.0, -1.0, -1.0, -1.0};
    for(int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            A.insert({i, j}, (float)c[i * 2 + j]);
        }
    }
    A.pack();

    Dropout<float> test_(1.0, 2, 2);
    test_.forward(B, A);

    std::cout << A << std::endl;
    std::cout << B << std::endl;
}

void test_layer_norm() { 
    taco::Tensor<float> A("A", {2, 2}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense});
    taco::Tensor<float> B("B", {2, 2}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense});
    std::vector<float> c{20.0, 8.0, 8.0, 2.0};
    for(int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            A.insert({i, j}, (float)c[i * 2 + j]);
        }
    }
    A.pack();

    LayerNorm<float> test_;
    test_.forward(B, A);

    std::cout << A << std::endl;
    std::cout << B << std::endl;
}

int main(int argc, char* argv[]) {
    int batch_size = 32;
    int max_seq = 512;
    int n_heads = 8;
    int n_key = 64;
    int n_model = n_heads * n_key;

    test_position();

    // taco::Tensor<float> Q("Q", {batch_size, max_seq, n_model}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense, taco::ModeFormat::Dense});
    // taco::Tensor<float> K("K", {batch_size, max_seq, n_model}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense, taco::ModeFormat::Dense});
    // taco::Tensor<float> V("V", {batch_size, max_seq, n_model}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense, taco::ModeFormat::Dense});

    // auto joined_heads = attention(Q, K, V, batch_size, max_seq, n_heads, n_model, n_key);

    return 0;
}