#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "common.h"
#include "layer.h"
#include "attention.h"
#include "layer_norm.h"
#include "dropout.h"
#include "feed_forward.h"

// using namespace taco;

template<typename T>
class TransformerBlock {
public:
	TransformerBlock(int embed_size, int heads, float dropout_prob,
					int forward_expansion) : attention(embed_size, heads), norm1(), norm2(),
											 feed_forward(embed_size, forward_expansion * embed_size, dropout_prob) ,
											 dropout(dropout_prob)
	{
	}
	taco::Tensor<T> operator()(const taco::Tensor<T> &value, const taco::Tensor<T> &key, const taco::Tensor<T> &query, const taco::Tensor<T> &mask) {
		int batch_size = value.getDimension(0);
		int max_seq = value.getDimension(1);
		int embed_size = value.getDimension(2);
		taco::Tensor<T> Att({batch_size, max_seq, embed_size}, dense);
		taco::Tensor<T> forward({batch_size, max_seq, embed_size}, dense);
		// TODO: Add mask for attention module
		attention.forward(Att, query, key, value);

		#ifdef DEBUG
		std::cout << Att << std::endl;
		std::cout << "After second attention";
		std::cin.ignore();
		#endif

		taco::Tensor<T> residual(query.getDimensions(), query.getFormat());
		taco::Tensor<T> norm(query.getDimensions(), query.getFormat());
		residual(i, j, k) = Att(i, j, k) + query(i, j, k);

		norm1.forward(norm, residual);
		taco::Tensor<T> x = dropout.forward(norm);
		feed_forward.forward(forward, x);
		#ifdef DEBUG
		feed_forward.print_weights();
		
		std::cout << "Press Enter to Continue: weights";
		std::cin.ignore();
		std::cout << forward << std::endl;
		std::cout << "Press Enter to Continue: forward";
		std::cin.ignore();
		#endif
		residual(i, j, k) = forward(i, j, k) + x(i, j, k);
		norm2.forward(norm, residual);
		taco::Tensor<T> out = dropout.forward(norm);

		return out;
	}

private:
    MultiHeadedAttention<T> attention;
    LayerNorm<T> norm1;
    LayerNorm<T> norm2;
    FeedForward<T> feed_forward;
	Dropout<T> dropout;
};

#endif //TRANSFORMER_LINEAR_H