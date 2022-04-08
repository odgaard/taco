#include "test.h"
#include "taco.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "linear.h"
#include "relu.h"
#include "softmax.h"
// #include "attention.h"
// #include "transformer_block.h"

#include "common.h"

// int batch_size = 32;
// int max_seq = 512;
// int n_heads = 8;
// int heads_dim = 64;
int batch_size = 8;
int max_seq = 64;
int n_heads = 4;
int heads_dim = 16;
int embed_size = n_heads * heads_dim;

struct format : public TestWithParam<ModeFormat> {};

TEST_P(format, reshape) {

    ModeFormat tnsFormat = GetParam();

    taco::Tensor<float> A("A", {2, 2, 4, 4}, tnsFormat);
    std::vector<float> c{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                         29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                         59, 60, 61, 62, 63, 64};
    int count = 0;
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            for(int k = 0; k < 4; k++) {
                for(int l = 0; l < 4; l++) {
                    A.insert({i, j, k, l}, (float)c[count]);
                    count++;
                }
            }
        }
    }
    A.pack();

    taco::Tensor<float> expected("expected", {4,4,4}, tnsFormat);

    count = 0;
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            for(int k = 0; k < 4; k++) {
                expected.insert({i, j, k}, (float)c[count]);
                count++;
            }
        }
    }
    expected.pack();

    taco::Tensor<float> actual = reshape(A, {4, 4, 4});
    ASSERT_TENSOR_EQ(expected, actual);
}

TEST_P(format, linear) {
    ModeFormat tnsFormat = GetParam();
    taco::Tensor<float> A("A", {4}, tnsFormat);
    taco::Tensor<float> actual("actual", {4}, tnsFormat);
    std::vector<float> c{20.0, 20.0, 20.0, 20.0};
    std::vector<float> expected_vec{16.0, 16.0, 16.0, 16.0};
    for(int i = 0; i < 4; i++) {
        // for (int j = 0; j < 8; j++) {
            A.insert({i}, (float)c[(i)]);
        // }
    }
    A.pack();

    taco::Tensor<float> expected("expected", {4}, dense);
    for(int i = 0; i < 4; i++) {
        // for (int j = 0; j < 8; j++) {
            expected.insert({i}, (float)expected_vec[(i)]);
        // }
    }

    Linear<float> linear(4, 4);
    linear.forward(actual, A);

    ASSERT_TENSOR_EQ(expected, actual);
}

TEST_P(format, RELU) {
    ModeFormat tnsFormat = GetParam();

    taco::Tensor<float> A("A", {1,2,2}, tnsFormat);
    taco::Tensor<float> actual("actual", {1,2,2}, tnsFormat);
    std::vector<float> c{20.0, -1.0, -1.0, 0.0};
    std::vector<float> expected_vec{20.0, 0.0, 0.0, 0.0};

    for(int i = 0; i < 1; i++) {
        for(int j = 0; j < 2; j++) {
            for(int k = 0; k < 2; k++) {
                A.insert({i, j, k}, (float)c[j * 2 + k]);
            }
        }
    }
    A.pack();

    taco::Tensor<float> expected("expected", {1,2,2}, tnsFormat);
    for(int i = 0; i < 1; i++) {
        for(int j = 0; j < 2; j++) {
            for(int k = 0; k < 2; k++) {
                expected.insert({i, j, k}, (float)expected_vec[j * 2 + k]);
            }
        }
    }
    expected.pack();

    ReLU<float> relu;
    relu.forward(actual, A);
    ASSERT_TENSOR_EQ(expected, actual);
}

TEST_P(format, Softmax) {
    ModeFormat tnsFormat = GetParam();

    taco::Tensor<float> A("A", {1,1,2,2}, tnsFormat);
    taco::Tensor<float> actual("actual", {1,1, 2,2}, tnsFormat);
    std::vector<float> c{20.0, 20.0, 20.0, 20.0};
    std::vector<float> expected_vec{0.5, 0.5, 0.5, 0.5};

    for(int l = 0; l < 1; l++)
        for(int i = 0; i < 1; i++) {
            for(int j = 0; j < 2; j++) {
                for(int k = 0; k < 2; k++) {
                    A.insert({l, i, j, k}, (float)c[j * 2 + k]);
                }
            }
        }
    A.pack();

    taco::Tensor<float> expected("expected", {1,1,2,2}, tnsFormat);

    for(int l = 0; l < 1; l++)
        for(int i = 0; i < 1; i++) {
            for(int j = 0; j < 2; j++) {
                for(int k = 0; k < 2; k++) {
                    expected.insert({l, i, j, k}, (float)expected_vec[j * 2 + k]);
                }
            }
        }
    expected.pack();

    Softmax<float> softmax;
    int last_dim = 3;
    softmax.forward(actual, A, last_dim);
    ASSERT_TENSOR_EQ(expected, actual);
}

INSTANTIATE_TEST_CASE_P(tensor, format, 
                        Values(dense, 
                               sparse));
