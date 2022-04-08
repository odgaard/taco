#ifndef TRANSFORMER_DECODER_BLOCK_H
#define TRANSFORMER_DECODER_BLOCK_H

#include "common.h"
#include "layer.h"
#include "layer_norm.h"
#include "transformer_block.h"
#include "attention.h"

// using namespace taco;

template<typename T>
class DecoderBlock {
public:
    DecoderBlock(int embed_size, int heads, 
                int forward_expansion, float dropout_prob) : layer_norm(), transformer_block(embed_size, heads, dropout_prob, forward_expansion),
                                                         dropout(dropout_prob), attention(embed_size, heads)
    {
    }
    taco::Tensor<T> operator()(const taco::Tensor<T> &x, const taco::Tensor<T> &value, const taco::Tensor<T> &key, 
                               const taco::Tensor<T> &src_mask, const taco::Tensor<T> &trg_mask) {
		int batch_size = value.getDimension(0);
		int max_seq = value.getDimension(1);
		int embed_size = value.getDimension(2);
		taco::Tensor<T> Att({batch_size, max_seq, embed_size}, x.getFormat());
        taco::Tensor<T> K(x.getDimensions(), x.getFormat());
        taco::Tensor<T> V(x.getDimensions(), x.getFormat());
        // K(i, j, k) = x(i, j, k);
        // V(i, j, k) = x(i, j, k);
        attention.forward(Att, x, x, x);
        taco::Tensor<T> residual(x.getDimensions(), x.getFormat());
        taco::Tensor<T> norm(x.getDimensions(), x.getFormat());
        residual(i, j, k) = Att(i, j, k) + x(i, j, k);

        // #ifdef DEBUG1
        // std::cout << residual << std::endl;
        // std::cout << "Residual (attention + x)";
        // std::cin.ignore();
        // #endif

        norm = layer_norm.forward(residual);

        #ifdef DEBUG1
        std::cout << norm << std::endl;
        std::cout << "Norm: layernorm(attention + x)";
        std::cin.ignore();
        #endif

        taco::Tensor<T> query = dropout.forward(norm);

        #ifdef DEBUG1
        std::cout << query << std::endl;
        std::cout << "query before transformer block";
        std::cin.ignore();

        std::cout << key << std::endl;
        std::cout << "key before transformer block";
        std::cin.ignore();

        std::cout << value << std::endl;
        std::cout << "value before transformer block";
        std::cin.ignore();
        #endif

        taco::Tensor<T> out = transformer_block(value, key, query, src_mask);

        #ifdef DEBUG
        std::cout << out << std::endl;
        std::cout << "After Decoder block";
        std::cin.ignore();
        #endif
        // std::cout << out << std::endl;
        // exit(1);
        return out;
    }

private:
    LayerNorm<T> layer_norm;
    TransformerBlock<T> transformer_block;
    Dropout<T> dropout;
    MultiHeadedAttention<T> attention;
};

#endif //TRANSFORMER_DECODER_BLOCK_H