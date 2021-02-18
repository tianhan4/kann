#include <cstdlib>
#include <cmath>
#include <iostream>
#include <omp.h>
#include "kann.h"
#include "kann_extra/kann_data.h"
#include "util.h"
#include "boost/asio.hpp"
#include "tcp/NetIO.hpp"
#include <sstream>

using namespace std;

void setup_engine(int batch_size, const string& base_dir)
{
    int i;
    string engine_path;
    cout << "set up the encryption engine" << endl;
    size_t poly_modulus_degree = 8192;
	size_t standard_scale = 24;
    std::vector<int> coeff_modulus = {30, 24, 24, 24, 30};
    SEALEncryptionParameters parms(poly_modulus_degree,
                    coeff_modulus,
                    seal_scheme::CKKS);
    engine = make_shared<SEALEngine>();

    engine_path = base_dir + "/" + ENGINE_FILE;
    if (load_engine(engine, engine_path) < 0) {
        cout << "laod engine fail in " << engine_path << endl;
        cout << "init engine from parms" << endl;
        engine->init(parms, standard_scale);
    }
	engine->max_slot() = batch_size;
    size_t slot_count = engine->slot_count();
    cout <<"Poly modulus degree: " << poly_modulus_degree<< endl;
    cout << "Coefficient modulus: ";
    cout << endl;
    cout << "slot count: " << slot_count<< endl;
    cout << "scale: 2^" << standard_scale << endl;
	// For multi-thread
    int max_parallel = omp_get_max_threads();
    cout << "max thread: " << max_parallel << endl;
	plaintext = new SEALPlaintext[max_parallel];
	for (i = 0; i < max_parallel; i ++){
		plaintext[i].init(engine);
	}
	ciphertext = new SEALCiphertext[max_parallel];
	for (i = 0; i < max_parallel; i ++){
		ciphertext[i].init(engine);
	}
	t = new vector<double>[max_parallel];
	for (i = 0; i < max_parallel; i ++){
		t[i].resize(engine->max_slot());
	}
	test_t = new vector<double>[max_parallel];
	for (i = 0; i < max_parallel; i ++){
		test_t[i].resize(engine->max_slot());
	}
	truth_t = new vector<double>[max_parallel];
	for (i = 0; i < max_parallel; i ++){
		truth_t[i].resize(engine->max_slot());
	}

	engine->zero = new SEALCiphertext(engine);
	engine->encode(0, *plaintext);
	engine->encrypt(*plaintext, *(engine->zero));
    //engine->zero = nullptr;
    engine->lazy_relinearization() = true;
    engine->lazy_mode() = true;    
}

int main(int argc, char *argv[])
{
    int i, j, k;
    int max_epoch = 4, batch_size = 12, max_drop_streak = 0;
    float lr = 0.15f, frac_val = 0.1f;
    int total_samples, left_sample_num;
    int data_size, label_size;
    kann_data_t *data, *label;
    kann_t *lenet;
    int batch_num, current_batch_size;
    string base_dir;

    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds training_time(0);

	vector<int> shuf;
    vector<vector<float>> truth;


    if (argc != 4) {
        cout << "Usage: " << argv[0] << " data" << " label" << "base_id:" << endl;
        return 0;
    }

    base_dir = string{argv[3]} + "/batch_" + to_string(batch_size);

    setup_engine(batch_size, base_dir);

    cout << "read data meta" << endl;
	data = kann_data_read(argv[1]);
    label = kann_data_read(argv[2]);

    if (data->n_row != label->n_row) {
        cout << "assert fail: data->n_row != label->n_row (" 
            << data->n_row << " != " << label->n_row << ")" << endl;
        // goto free_data;
    }

    //total_samples = data->n_row;
    total_samples = 12;
    data_size = data->n_col;
    label_size = label->n_col;

    shuf.reserve(total_samples);
    for (i = 0; i < total_samples; ++i) shuf[i] = i;
    // batch_num = total_samples % batch_size == 0? total_samples / batch_size : total_samples / batch_size + 1;
    batch_num = shuffle_and_encrypt_dataset(total_samples, batch_size, data, label, base_dir, shuf);
    cout << "batch_num : " << batch_num << endl; 
    if (batch_num < 0) {
        cout << "[error] batch num (" << batch_num << ") < 0" << endl;
        // goto free_data;
    }

    truth.resize(batch_num);
    for (i = 0, left_sample_num = total_samples; i < batch_num; ++i, left_sample_num -= current_batch_size) {
        current_batch_size = left_sample_num > batch_size ? batch_size : left_sample_num;
        truth[i].reserve(current_batch_size*label_size);
        for (j = 0; j < current_batch_size; ++j) {
            for (k = 0; k < label_size; ++k) {
                truth[i][j*label_size+k] = label->x[i*batch_size+j][k];
            }
        }
    }

    kann_data_free(data);
    kann_data_free(label);

    cout << "save encrypted ciphertext in " << base_dir << endl;

    //=======

/**
	int ret_size = 0;
	string batch_dir = base_dir + "/0";
	vector<SEALCiphertext> _x(data_size); 
    vector<SEALCiphertext> _y(label_size);
	SEALCiphertext *x_ptr, *y_ptr;
	// 1. load the data/label
	ret_size = load_batch_ciphertext(_x, data_size, batch_dir, 0);
	if (ret_size != data_size) {
		cout << "[train ] load data return " << ret_size << ". expect " << data_size << endl;
		return;
	}
	ret_size = load_batch_ciphertext(_y, label_size, batch_dir, 1);
	if (ret_size != label_size) {
		cout << "[train ] load label return " << ret_size << ". expect " << label_size << endl;
		return;
	}
	x_ptr = _x.data(), y_ptr = _y.data();
**/
    int op, dim[5], dim_size;
    vector<SEALCiphertext> ciphertext_pool;  
    cout << "trying to connect!..." << endl;
    NetIO network_io("127.0.0.1", 17722, engine);
    cout << "connected!" << endl;
    bool flag = true;

    while(flag){
/*
        
        ifstream ct;
        SEALCiphertext tmp(engine);
        ct.open("tmp.save", ios::binary);
        for (int i=0;i<30;i++){
            char b;
            ct.read(&b,1);
            cout << i << ": " << int(b) << endl;
        }
        //print_ciphertext(&tmp);
        ct.close();
*/


        cout << "finished" << endl;
        int number = network_io.recv_ciphertext(&op, ciphertext_pool, dim, &dim_size);
        cout << "ciphertext received" << endl;
        for(int i=0;i<number;i++)
            print_ciphertext(&ciphertext_pool[i]);
        //do something
        switch (op)
        {
        case remote_ops::OP_RELU_FP:
            //do relu
            #pragma omp parallel
            {
            #pragma omp for
                for (i = 0; i < number; i++){
                    int thread_mod_id = omp_get_thread_num()%omp_get_max_threads();
                    engine->decrypt(ciphertext_pool[i], plaintext[thread_mod_id]);
                    engine->decode(plaintext[thread_mod_id], t[thread_mod_id]);
                    for (j = 0; j < t[thread_mod_id].size(); ++j)
                        t[thread_mod_id][j] = t[thread_mod_id][j] > 0.0f? t[thread_mod_id][j] : 0.0f;
                    engine->encode(t[thread_mod_id], plaintext[thread_mod_id]);
                    engine->encrypt(plaintext[thread_mod_id], ciphertext_pool[i]);
                }
            }
            for(int i=0;i<number;i++)
                print_ciphertext(&ciphertext_pool[i]);
            network_io.send_ciphertext(op, ciphertext_pool.data(), number, dim, dim_size);
            break;
        
        default:
            cout << "unknown op!" << endl;
            flag = false;
            break;
        }

    }

    kann_delete(lenet);
    delete[] plaintext;
	delete[] ciphertext;
	delete[] t;
	delete[] truth_t;
	delete[] test_t;
    return 0;
free_data:
    delete[] plaintext;
	delete[] ciphertext;
	delete[] t;
	delete[] truth_t;
	delete[] test_t;
    kann_data_free(data);   
    kann_data_free(label);
    return 0;
}
