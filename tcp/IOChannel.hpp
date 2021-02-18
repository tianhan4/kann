#pragma once

#include <memory>
#include "SEALEngine.h"

enum remote_ops{
	OP_SIGM_FP,
	OP_TANH_FP,
	OP_MAX2D_FP,
	OP_CE_BIN_FP,
	OP_CE_MULTI_FP,
	OP_LOG_FP,
	OP_RELU_FP,
	OP_MSE_FP,
	OP_SOFTMAX_FP,
	OP_AVG_FP,
	OP_CE_BIN_NEG_FP,
	OP_SIGM_BP,
	OP_TANH_BP,
	OP_MAX2D_BP,
	OP_CE_BIN_BP,
	OP_CE_MULTI_BP,
	OP_LOG_BP,
	OP_RELU_BP,
	OP_MSE_BP,
	OP_SOFTMAX_BP,
	OP_AVG_BP,
	OP_CE_BIN_NEG_BP,
	OP_DP_DECRYPTION
};

template<typename T> 
class IOChannel { public:
	uint64_t counter = 0;
	void send_data(const void * data, int nbyte) {
		counter +=nbyte;
		derived().send_data_internal(data, nbyte);
	}

	void recv_data(void * data, int nbyte) {
		derived().recv_data_internal(data, nbyte);
	}

	void send_ciphertext(int op, SEALCiphertext* ciphertext, int number, int *dim, int dim_size) {
		//0.send op
		//1.send dim number
		//2.send dim
		//3.send ciphertext number
		//4.send ciphertext
		send_data(&op, sizeof(int));
		send_data(&dim_size, sizeof(int));
		send_data(dim, sizeof(int)*dim_size);
		send_data(&number, sizeof(int));
		int total = 1;
		for (int j = 0; j < dim_size; j++)
			total *= dim[j];
		assert(total == number);
		for (int i = 0; i< number; i++){
			derived().send_ciphertext_internal(ciphertext+i);
		}
	}

	size_t recv_ciphertext(int *op, SEALCiphertext* ciphertext, int *dim, int *dim_size) {
		//0.recv op
		//1.recv dim number
		//2.recv dim
		//3.recv ciphertext number
		//4.recv ciphertext
		int number;
		recv_data(op, sizeof(int));
		recv_data(dim_size, sizeof(int));
		recv_data(dim, sizeof(int)*(*dim_size));
		recv_data(&number, sizeof(int));
		cout << *op << endl;
		cout << *dim_size << endl;
		for (int i = 0; i< *dim_size; i++)
			cout << dim[i];
		cout << endl;
		cout << number << endl;
		int total = 1;
		for (int j = 0; j < *dim_size; j++)
			total *= dim[j];
		assert(total == number);
		for (int i = 0; i< number; i++){
			derived().recv_ciphertext_internal(ciphertext+i);
		}
		return number;
	}

	size_t recv_ciphertext(int *op, vector<SEALCiphertext> & ciphertext_vector, int *dim, int *dim_size) {
		//0.recv op
		//1.recv dim number
		//2.recv dim
		//3.recv ciphertext number
		//4.recv ciphertext
		int number;
		recv_data(op, sizeof(int));
		recv_data(dim_size, sizeof(int));
		recv_data(dim, sizeof(int)*(*dim_size));
		recv_data(&number, sizeof(int));
		cout << *op << endl;
		cout << *dim_size << endl;
		for (int i = 0; i< *dim_size; i++)
			cout << dim[i];
		cout << endl;
		cout << number << endl;
		int count = 0;
		
		/*
		//check some bytes
		for(int i=0;i<30;i++){
			char b;
			recv_data(&b, 1);
			cout << i << ":" << int(b) <<  endl;
		}*/
		

		int total = 1;
		for (int j = 0; j < *dim_size; j++)
			total *= dim[j];
		assert(total == number);
		//number = 2;
		//derived().recv_ciphertext_internal(&tmp);
		while(ciphertext_vector.size() < number){
			SEALCiphertext tmp(engine);
			ciphertext_vector.push_back(tmp);
		}
		for (int i = 0; i< number; i++){
			derived().recv_ciphertext_internal(&ciphertext_vector[i]);
		}
		return number;
	}

	void send_plaintext(float* values, int number, int *dim, int dim_size) {
		//1.send dim number
		//2.send dim
		//3.send raw value number
		//4.send values
		send_data(&dim_size, sizeof(int));
		send_data(dim, sizeof(int)*dim_size);
		send_data(&number, sizeof(int));
		int total = 1;
		for (int j = 0; j < dim_size; j++)
			total *= dim[j];
		assert(total == number);
		send_data(values, number*sizeof(float));
	}

	size_t recv_plaintext(float* values, int *dim, int *dim_size) {
		//1.recv dim number
		//2.recv dim
		//3.recv raw value number
		//4.recv values
		int number;
		recv_data(dim_size, sizeof(int));
		recv_data(dim, sizeof(int)*(*dim_size));
		recv_data(&number, sizeof(int));
		int total = 1;
		for (int j = 0; j < *dim_size; j++)
			total *= dim[j];
		assert(total == number);
		recv_data(values, number*sizeof(float));
		return number;
	}

	private:
	T& derived() {
		return *static_cast<T*>(this);
	}
};