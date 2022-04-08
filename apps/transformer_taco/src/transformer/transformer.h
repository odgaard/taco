#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "layer.h"
#include "common.h"
#include "softmax.h"
#include "feed_forward.h"
#include "layer_norm.h"
#include "dropout.h"
#include "decoder.h"
#include "decoder_block.h"
#include "module_list.h"

template<typename T>
class Transformer {
public: 
    Transformer(int src_vocab_size, int trg_vocab_size, int src_pad_idx,
                int trg_pad_idx, int embed_size=512, int num_layers=6,
                int forward_expansion=4, int heads=8, 
                T dropout=0.0f, int max_length=100) : decoder(trg_vocab_size, embed_size, num_layers,
                                                              heads, forward_expansion, dropout, max_length),
                                                      src_pad_idx(src_pad_idx), trg_pad_idx(trg_pad_idx)
    {

    }

    taco::Tensor<T> operator()(taco::Tensor<T> src, taco::Tensor<T> trg) {
        int N = src.getDimension(0);
        int src_len = src.getDimension(1);
        int trg_len = trg.getDimension(1);
        taco::Tensor<T> src_mask({N, 1, 1, src_len});
        taco::Tensor<T> trg_mask({N, 1, trg_len, trg_len});
        taco::Tensor<T> out = decoder(trg, src, src_mask, trg_mask);
        return out;
    }


private:
    Decoder<T> decoder;
    int src_pad_idx;
    int trg_pad_idx;
};

#endif // TRANSFORMER_H