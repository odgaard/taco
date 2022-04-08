// #include <iostream>
// #include <algorithm>
// #include <memory>

// #include "attention.h"
// #include "common.h"
// #include "relu.h"
// #include "dropout.h"
// #include "layer_norm.h"
// #include "position_encoding.h"
// #include "feed_forward.h"
// #include "transformer_block.h"
// #include "decoder.h"
// #include <xtensor/xadapt.hpp>
// #include <xtensor/xarray.hpp>
// #include <xtensor/xtensor.hpp>
// #include <xtensor/xbuffer_adaptor.hpp>

// // int batch_size = 32;
// // int max_seq = 512;
// // int n_heads = 8;
// // int heads_dim = 64;
// int batch_size = 8;
// int max_seq = 64;
// int n_heads = 4;
// int heads_dim = 16;
// int embed_size = n_heads * heads_dim;

// void test_decoder() {
//     typedef float Type;
//     taco::Tensor<Type> x("Q", {batch_size, max_seq, embed_size}, dense);
//     // taco::Tensor<float> K("K", {batch_size, max_seq, embed_size}, dense);
//     // taco::Tensor<float> V("V", {batch_size, max_seq, embed_size}, dense);
//     taco::Tensor<Type> Att("Att", {batch_size, max_seq, embed_size}, dense);

//     int trg_vocab_size = 8;
//     std::vector<float> val{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
//     // for(int l = 0; l < src_vocab_size; l++) {
//         int count = 0;
//         for(int i = 0; i < batch_size; i++) {
//             for(int j = 0; j < max_seq; j++) {
//                 for(int k = 0; k < embed_size; k++) {
//                     x.insert({i, j, k}, (Type)2.0);
//                     // x.insert({i, j, k}, (Type)val[count % 10]);
//                     count++;
//                 }
//             }
//         }
//     // }

//     x.pack();

//     Decoder<Type> decoder(trg_vocab_size, embed_size, 6, n_heads, 4, 0, max_seq);
//     taco::Tensor<Type> test = decoder(x, x, Att, Att);
//     std::cout << test << std::endl;
// }

// void test_transformer_block() {
//     taco::Tensor<float> Q("Q", {batch_size, max_seq, embed_size}, dense);
//     taco::Tensor<float> K("K", {batch_size, max_seq, embed_size}, dense);
//     taco::Tensor<float> V("V", {batch_size, max_seq, embed_size}, dense);
//     taco::Tensor<float> Att("Att", {batch_size, max_seq, embed_size}, dense);
    
//     for(int i = 0; i < batch_size; i++) {
//         for(int j = 0; j < max_seq; j++) {
//             for(int k = 0; k < embed_size; k++) {
//                 Q.insert({i, j, k}, (float)12.0);
//                 K.insert({i, j, k}, (float)13.0);
//                 V.insert({i, j, k}, (float)14.0);
//                 // Q.insert({i, j, k}, (float)c[(k + embed_size *(i * max_seq + j)) % 8]);
//                 // K.insert({i, j, k}, (float)c[(k + embed_size *(i * max_seq + j)) % 8]);
//                 // V.insert({i, j, k}, (float)c[(k + embed_size *(i * max_seq + j)) % 8]);
//             }
//         }
//     }

//     Q.pack();
//     K.pack();
//     V.pack();

//     TransformerBlock<float> transformer_block(embed_size, n_heads, 0.1, 4);
//     taco::Tensor<float> test = transformer_block(Q, K, V, Att);
//     std::cout << test << std::endl;
// }

// void test_attention() {
//     taco::Tensor<float> Q("Q", {batch_size, max_seq, embed_size}, dense);
//     taco::Tensor<float> K("K", {batch_size, max_seq, embed_size}, dense);
//     taco::Tensor<float> V("V", {batch_size, max_seq, embed_size}, dense);
//     taco::Tensor<float> Att("Att", {batch_size, max_seq, embed_size}, dense);
    
//     for(int i = 0; i < batch_size; i++) {
//         for(int j = 0; j < max_seq; j++) {
//             for(int k = 0; k < embed_size; k++) {
//                 Q.insert({i, j, k}, (float)12.0);
//                 K.insert({i, j, k}, (float)13.0);
//                 V.insert({i, j, k}, (float)14.0);
//                 // Q.insert({i, j, k}, (float)c[(k + embed_size *(i * max_seq + j)) % 8]);
//                 // K.insert({i, j, k}, (float)c[(k + embed_size *(i * max_seq + j)) % 8]);
//                 // V.insert({i, j, k}, (float)c[(k + embed_size *(i * max_seq + j)) % 8]);
//             }
//         }
//     }

//     Q.pack();
//     K.pack();
//     V.pack();

//     MultiHeadedAttention<float> attention(embed_size, n_heads);
//     attention.forward(Att, Q, K, V);
//     std::cout << Att << std::endl;
// }

// void test_reshape() {
//     taco::Tensor<float> A("A", {2, 2, 4, 4}, dense);
//     std::vector<float> c{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
//                          29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
//                          59, 60, 61, 62, 63, 64};
//     // c.resize()
//     int count = 0;
//     for(int i = 0; i < 2; i++) {
//         for(int j = 0; j < 2; j++) {
//             for(int k = 0; k < 4; k++) {
//                 for(int l = 0; l < 4; l++) {
//                     A.insert({i, j, k, l}, (float)c[count]);
//                     count++;
//                 }
//             }
//         }
//     }
//     A.pack();
//     std::cout << A << std::endl;

//     taco::Tensor<float> test = reshape(A, {4, 4, 4});
// }

// void test_relu() { 
//     taco::Tensor<float> A("A", {2, 2}, {dense, dense});
//     taco::Tensor<float> B("B", {2, 2}, {dense, dense});
//     std::vector<float> c{20.0, -1.0, -1.0, -1.0};
//     for(int i = 0; i < 2; i++) {
//         for (int j = 0; j < 2; j++) {
//             A.insert({i, j}, (float)c[i * 2 + j]);
//         }
//     }
//     A.pack();

//     ReLU<float> test_;
//     test_.forward(B, A);

//     std::cout << A << std::endl;
//     std::cout << B << std::endl;
// }

// void test_position() { 
//     taco::Tensor<float> A("A", {4, 4}, {dense, dense});
//     taco::Tensor<float> B("B", {4, 4}, {dense, dense});
//     std::vector<float> c{20.0, -1.0, -1.0, -1.0, 1.0, 1.0, 10.0, 10.0, 5.0, 5.0, 5.0, .5, .5, .5, .5, .5};
//     for(int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4; j++) {
//             A.insert({i, j}, (float)c[i * 4 + j]);
//         }
//     }
//     A.pack();

//     PositionEncoding<float> test_(4, 4);
//     test_.forward(B, A);

//     // printTensor(B);

//     // std::cout << A << std::endl;
//     // std::cout << B << std::endl;
// }

// void test_softmax() { 
//     taco::Tensor<float> A("A", {2, 2}, {dense, dense});
//     taco::Tensor<float> B("B", {2, 2}, {dense, dense});
//     std::vector<float> c{20.0, -1.0, -1.0, -1.0};
//     for(int i = 0; i < 2; i++) {
//         for (int j = 0; j < 2; j++) {
//             A.insert({i, j}, (float)c[i * 2 + j]);
//         }
//     }
//     A.pack();

//     Softmax<float> test_;
//     test_.forward(B, A);

//     std::cout << A << std::endl;
//     std::cout << B << std::endl;
// }

// void test_dropout() { 
//     taco::Tensor<float> A("A", {8, 8}, {dense, dense});
//     taco::Tensor<float> B("B", {8, 8}, {dense, dense});
//     std::vector<float> c{20.0, -1.0, -1.0, -1.0};
//     for(int i = 0; i < 8; i++) {
//         for (int j = 0; j < 8; j++) {
//             A.insert({i, j}, (float)c[(i * 2 + j) % 4]);
//         }
//     }
//     A.pack();

//     Dropout<float> test_(0.8);
//     test_.forward(B, A);

//     // printTensor(B);
// }

// void test_feedforward() { 
//     taco::Tensor<float> A("A", {50}, dense);
//     taco::Tensor<float> B("B", {50}, dense);
//     std::vector<float> c{20.0, 100.0, 1.0, 10.0};
//     for(int i = 0; i < 50; i++) {
//         // for (int j = 0; j < 8; j++) {
//             A.insert({i}, (float)c[(i) % 4]);
//         // }
//     }
//     A.pack();

//     FeedForward<float> test_(50, 50, 0.4);
//     test_.forward(B, A);

//     std::cout << A << std::endl;
//     std::cout << B << std::endl;
//     // printTensor(A);
//     // printTensor(B);
// }

// void test_linear() { 
//     taco::Tensor<float> A("A", {50}, dense);
//     taco::Tensor<float> B("B", {50}, dense);
//     std::vector<float> c{20.0, 100.0, 1.0, 10.0};
//     for(int i = 0; i < 50; i++) {
//         // for (int j = 0; j < 8; j++) {
//             A.insert({i}, (float)c[(i) % 4]);
//         // }
//     }
//     A.pack();

//     Linear<float> test_(50, 50);
//     test_.forward(B, A);

//     std::cout << A << std::endl;
//     std::cout << B << std::endl;
//     // printTensor(A);
//     // printTensor(B);
// }

// void test_layer_norm() { 
//     taco::Tensor<float> A("A", {2, 2}, {dense, dense});
//     taco::Tensor<float> B("B", {2, 2}, {dense, dense});
//     std::vector<float> c{20.0, 8.0, 8.0, 2.0};
//     for(int i = 0; i < 2; i++) {
//         for (int j = 0; j < 2; j++) {
//             A.insert({i, j}, (float)c[i * 2 + j]);
//         }
//     }
//     A.pack();

//     LayerNorm<float> test_;
//     test_.forward(B, A);

//     std::cout << A << std::endl;
//     std::cout << B << std::endl;
// }

// int main(int argc, char* argv[]) {
//     // test_linear();
//     // test_attention();
//     // test_reshape();

//     // test_transformer_block();
//     // taco::taco_set_num_threads(64);
//     test_decoder();

//     // taco::Tensor<float> test("test", {32,8,64,512}, dense);

//     // test.transpose({0,2,1,3});

//     // taco::Tensor<float> Q("Q", {batch_size, max_seq, n_model}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense, taco::ModeFormat::Dense});
//     // taco::Tensor<float> K("K", {batch_size, max_seq, n_model}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense, taco::ModeFormat::Dense});
//     // taco::Tensor<float> V("V", {batch_size, max_seq, n_model}, taco::Format{taco::ModeFormat::Dense, taco::ModeFormat::Dense, taco::ModeFormat::Dense});

//     // auto joined_heads = attention(Q, K, V, batch_size, max_seq, n_heads, n_model, n_key);

//     return 0;
// }