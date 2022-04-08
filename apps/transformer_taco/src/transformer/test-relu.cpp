#include "test.h"
#include "taco.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
// #include "attention.h"
// #include "transformer_block.h"

// struct format : public TestWithParam<ModeFormat> {};

// TEST_P(format, tensor) {
//     ModeFormat tnsFormat = GetParam();

//     taco::Tensor<float> A("A", {2,2}, tnsFormat);
//     taco::Tensor<float> actual("actual", {2,2}, tnsFormat);
//     std::vector<float> c{20.0, -1.0, -1.0, 0.0};
//     std::vector<float> expected_vec{20.0, 0.0, 0.0, 0.0};

//     for(int i = 0; i < 2; i++) {
//         for(int j = 0; j < 2; j++) {
//             A.insert({i, j}, (float)c[i * 2 + j]);
//         }
//     }
//     A.pack();

//     taco::Tensor<float> expected("expected", {2,2}, tnsFormat);
//     for(int i = 0; i < 2; i++) {
//         for(int j = 0; j < 2; j++) {
//             expected.insert({i, j}, (float)expected_vec[i * 2 + j]);
//         }
//     }
//     expected.pack();

//     ReLU<float> relu;
//     relu.forward(actual, A);
//     ASSERT_TENSOR_EQ(expected, actual);
// }

// INSTANTIATE_TEST_CASE_P(tensor, format, 
//                         Values(dense, 
//                                sparse));