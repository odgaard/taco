#ifndef TRANSFORMER_DECODER_H
#define TRANSFORMER_DECODER_H

#include "common.h"
#include "layer.h"
#include "position_encoding.h"
#include "decoder_block.h"
#include "dropout.h"
#include "linear.h"
#include "module_list.h"

// using namespace taco;

template<typename T>
class Decoder {
public:
    Decoder(int trg_vocab_size, int embed_size, int num_layers,
            int heads, int forward_expansion, float dropout_prob, int max_seq) : position_embedding(max_seq, embed_size), fc_out(embed_size, trg_vocab_size, false),
                                                                                 dropout(dropout_prob), module_list(num_layers)
    {
        // TODO: Use torch C++ API for word embedding
        for(int i = 0; i < num_layers; ++i) {
            DecoderBlock<T> decoder_block(embed_size, heads, forward_expansion, dropout_prob);
            module_list.push(decoder_block);
        }
    }
    taco::Tensor<T> operator()(const taco::Tensor<T> &x, const taco::Tensor<T> &enc_out, 
                               taco::Tensor<T> &src_mask, taco::Tensor<T> &trg_mask) {
        // TODO: Either pass in x as the word embedding itself or as the actual input
        taco::Tensor<T> emb_with_pos = position_embedding.forward(x);
        taco::Tensor<T> temp_x = dropout.forward(emb_with_pos);
        std::string name = x.getName();
        // x = temp_x;
        // x.setName(name);
        // std::cout << x << std::endl;
        // std::cout << "Press Enter to Continue: Decoder";
        // std::cin.ignore();
        taco::Tensor<T> enc = temp_x;
        taco::Tensor<T> temp = module_list(temp_x, enc, enc, src_mask, trg_mask);

        // fc_out.print_linear_dim();
        #ifdef DEBUG1
        shape(temp, "Temp");
        #endif
        std::cout << temp << std::endl;
        shape(fc_out.get_weights(), "Weights");
        std::cout << "out before fc out";
        std::cin.ignore();

        shape(temp, "before linear");
        taco::Tensor<T> out = fc_out.forward(temp);
        shape(out, "after linear");

        taco::Tensor<T> slicer({1, 1, 8}, dense);
        taco::Tensor<T> input_copy = out;
        int i_ = 0;
        int j_ = 4;
        slicer(i, j, k) = input_copy(i(i_, i_+1), j(j_, j_+1), k);

        std::cout << slicer << std::endl;
        std::cout << "Out slice";
        std::cin.ignore();

        return out;
    }
private:
    PositionEncoding<T> position_embedding;
    Linear<T> fc_out;
    Dropout<T> dropout;
    ModuleList<DecoderBlock<T>, T> module_list;
};

#endif //TRANSFORMER_DECODER_H