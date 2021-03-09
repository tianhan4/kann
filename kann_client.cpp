#include <cstdlib>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <cfloat>
#include "NetIO.h"
#include "kann.h"
#include "kann_extra/kann_data.h"
#include "util.h"
#include <sstream>
#include <random>

using namespace std;


double noise_multiplier = 1.1;
double l2_norm_clip = 1.0;

const double mean = 0.0;
std::default_random_engine random_generator;
std::normal_distribution<double> dist(mean, l2_norm_clip * noise_multiplier);


#define conv_out_size(in_size, aux) (((in_size) - (aux)->kernel_size + (aux)->pad[0] + (aux)->pad[1]) / (aux)->stride + 1)

#define conv_in_size(out_size, aux) (((out_size) - 1) * (aux)->stride - (aux)->pad[0] - (aux)->pad[1] + (aux)->kernel_size)

#define process_row_for(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
	int j, l; \
	if (_stride > 1) { \
		for (l = 0; l < _wn; ++l) { \
			const SEALCiphertext *xl = &_xx[l - _pad]; \
			for (j = 0; j < _pn; ++j, xl += _stride) _t[j] = *xl; \
			kad_saxpy(_pn, _ww[l], _t, _yy); \
		} \
	} else for (l = 0; l < _wn; ++l) { \
		kad_saxpy(_pn, _ww[l], &_xx[l - _pad], _yy); \
	} \
} while (0)

#define process_row_back_x(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
	int j, l, k; \
	if (_stride > 1) { \
		for (l = 0; l < _wn; ++l) { \
			SEALCiphertext *xl = &_xx[l - _pad]; \
            for (k = 0; k < _pn; k++) \
                _t[k].clean() = true; \
			kad_saxpy(_pn, _ww[l], _yy, _t); \
			for (j = 0; j < _pn; ++j, xl += _stride) seal_add_inplace(*xl, _t[j]); \
		} \
	} else for (l = 0; l < _wn; ++l) kad_saxpy(_pn, _ww[l], _yy, &_xx[l - _pad]); \
} while (0)

#define process_row_back_w(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
	int j, l; \
	if (_stride > 1) { \
		for (l = 0; l < _wn; ++l) { \
			const SEALCiphertext *xl = &_xx[l - _pad]; \
			for (j = 0; j < _pn; ++j, xl += _stride) _t[j] = *xl; \
			*ciphertext = kad_sdot(_pn, _yy, _t); \
            seal_add_inplace(_ww[l], *ciphertext); \
		} \
	} else for (l = 0; l < _wn; ++l){ \
        *ciphertext = kad_sdot(_pn, _yy, &_xx[l - _pad]); \
        seal_add_inplace(_ww[l], *ciphertext); \
    } \
} while (0)



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
    //base_dir = "/data/glusterfs/home/htianab/kann-data/engine_8192_30_24_24_24_30";
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
    int op, dim[5], dim_size, layer;
    int op2, dim2[5], dim_size2;
    vector<SEALCiphertext> ciphertext_pool; 
    vector<SEALCiphertext> ciphertext_pool2; 
    vector<vector<vector<double>>> gradients; // (layer, dimension) gradients
    vector<vector<double>> gradients4dp;
    vector<float> processed_gradients;
    cout << "trying to connect!..." << endl;
    NetIO network_io("127.0.0.1", 17722, engine);
    cout << "connected!" << endl;
    bool flag = true;
    vector<conv_conf_t*> maxIndex;
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
        cout << "new task coming!" << endl;
        int number = network_io.recv_ciphertext(&op, &layer, ciphertext_pool, dim, &dim_size);
        cout << op << "op" << endl;
        cout << number << "nunmber" << endl;
        int batch_size = ciphertext_pool[0].size();
        //for(int i=0;i<number;i++)
        //    print_ciphertext(&(ciphertext_pool.data()[i]));

		while(ciphertext_pool2.size() < number){
			SEALCiphertext tmp(engine);
			ciphertext_pool2.push_back(tmp);
		}
		while(gradients.size() <= layer){
			gradients.push_back(vector<vector<double>>());
		}
        while(gradients[layer].size() < number){
            gradients[layer].push_back(vector<double>());
        }
        //only when FP
        if(is_fp(op))
            for(int i = 0 ; i < number ; i ++)
                gradients[layer][i].resize(batch_size);

        while(maxIndex.size() <= layer){
            maxIndex.push_back(nullptr);
        }
        //do something
        //the logic: the server send gradients for the following layer + some other information(maybe ciphers)
        //           the client send the gradients for the current layer back.
        //           ps: the client may store some gradient information during the forwarding.
        //           ps: the batch_size is always obtained dynamically.
        switch (op){
            case remote_ops::OP_RELU_FP:{

                //do relu
                #pragma omp parallel
                {
                #pragma omp for
                    //cache the plaintext using test_t
                    for (i = 0; i < number; i++){
                        int thread_mod_id = omp_get_thread_num()%omp_get_max_threads();
                        engine->decrypt(ciphertext_pool[i], plaintext[thread_mod_id]);
                        engine->decode(plaintext[thread_mod_id], t[thread_mod_id]);
                        for (j = 0; j < t[thread_mod_id].size(); ++j){
                            t[thread_mod_id][j] = t[thread_mod_id][j] > 0.0f? t[thread_mod_id][j] : 0.0f;
                            gradients[layer][i][j] = t[thread_mod_id][j] >  1e-4? 1.0f: 0.0f;
                        }
                        engine->encode(t[thread_mod_id], plaintext[thread_mod_id]);
                        engine->encrypt(plaintext[thread_mod_id], ciphertext_pool[i]);
                    }
                }
                network_io.send_ciphertext(op, layer, ciphertext_pool.data(), dim, dim_size);
                break;
            }
            case remote_ops::OP_RELU_BP:{
                #pragma omp parallel
                {
                #pragma omp for
                    for (i=0; i< number; i++){
                        int thread_mod_id = omp_get_thread_num()%omp_get_max_threads();
                        if(ciphertext_pool[i].clean()){
                            continue;
                        }else{
                            engine->decrypt(ciphertext_pool[i], plaintext[thread_mod_id]);
                            engine->decode(plaintext[thread_mod_id], t[thread_mod_id]);
                            for (j = 0; j < t[thread_mod_id].size(); ++j)
                                t[thread_mod_id][j] = t[thread_mod_id][j] * gradients[layer][i][j];
                            engine->encode(t[thread_mod_id], plaintext[thread_mod_id]);
                            engine->encrypt(plaintext[thread_mod_id], ciphertext_pool[i]);		
                        }
                    }
                }
                network_io.send_ciphertext(op, layer, ciphertext_pool.data(), dim, dim_size);
                break;
            }
            case remote_ops::OP_CE_MULTI_FP:{
                int number2 = network_io.recv_ciphertext(&op2, &layer, ciphertext_pool2, dim2, &dim_size2);
                
                vector<double> cost(batch_size, 0.);
                for (i = 0; i < number; ++i){
                    engine->decrypt(ciphertext_pool[i], *plaintext);
                    engine->decode(*plaintext, test_t[0]);
                    engine->decrypt(ciphertext_pool2[i], *plaintext);
                    engine->decode(*plaintext, truth_t[0]);
                    for (j=0; j<batch_size;j++){
                        cost[j] += -log(test_t[0][j]<=1e-4?1e-4:test_t[0][j])*(truth_t[0][j]<1e-4?0:truth_t[0][j])/(float)(batch_size);
                        gradients[layer][i][j] = -1.0 * (truth_t[0][j]<1e-4?0.:truth_t[0][j]) / (test_t[0][j]<=1e-4?1e-4:test_t[0][j]);
                    }
                }
                engine->encode(cost, *plaintext);
                engine->encrypt(*plaintext, ciphertext_pool[0]);
                dim_size = 1;
                dim[0] = 1;
                network_io.send_ciphertext(op, layer, ciphertext_pool.data(), dim, 1);
                break;
            }

            case remote_ops::OP_CE_MULTI_BP:{
                number = gradients[layer].size();
                int new_batch_size = gradients[layer][0].size();
                engine->decrypt(ciphertext_pool[0], *plaintext);
                engine->decode(*plaintext, t[0]);
                double coeff = t[0][0]/new_batch_size;
                t[0].resize(new_batch_size);
                //cout << "coeff:" << coeff << endl;
                for (i = 0; i < number; ++i){
                    for (j = 0; j < new_batch_size; ++j)
                        t[0][j] = coeff * gradients[layer][i][j];
                    engine->encode(t[0], *plaintext);
                    engine->encrypt(*plaintext, ciphertext_pool[i]);
                }
                dim[0] = number;
                dim_size = 1;
                network_io.send_ciphertext(op, layer, ciphertext_pool.data(), dim, 1);
                break;
            }
            case remote_ops::OP_MAX2D_FP:{
                conv_conf_t *aux = new conv_conf_t[2];
                network_io.recv_data(aux, 2*sizeof(conv_conf_t));
                assert((aux[0].pad[0] == 0 && aux[0].pad[1] == 0 && aux[1].pad[0] == 0 && aux[1].pad[1] == 0));
                assert(aux[0].stride==aux[0].kernel_size && aux[1].stride==aux[1].kernel_size);
                assert(dim_size == 3);
                assert(dim[1]%aux[0].stride==0);
                assert(dim[2]%aux[1].stride==0);
                int output_number = 1, output_dim_size;
                int output_dim[5];
                //1. kad_sync_dim is needed to get the output tensor size.
                if (dim_size != 3) assert(false);
                output_dim_size = 3;
                output_dim[0] = dim[0], output_dim[1] = conv_out_size(dim[1], &aux[0]), output_dim[2] = conv_out_size(dim[2], &aux[1]);
                for(int i=0; i< output_dim_size; i++)
                    output_number *= output_dim[i];
                //2. standard forward process
                int rest = 1, len, i;
                //transform gtmp into gradients structure.
                len = output_number;

                for (i = 0; i < len; ++i) ciphertext_pool2[i].clean() = true;
                for (i = 0; i < output_dim_size - 2; ++i) rest *= output_dim[i];
                int p_row = output_dim[output_dim_size - 2], p_col = output_dim[output_dim_size - 1];
                int full_ii = rest * p_row * p_col;
                #pragma omp parallel
                {
                #pragma omp for
                for (int ii = 0; ii < full_ii; ii++){
                    int t = ii / (p_row * p_col);
                    int i = (ii / p_col) % (p_row);
                    int j = (ii % p_col);
                    int thread_mod_id = omp_get_thread_num()%omp_get_max_threads();
                    //for (t = 0; t < rest; ++t) {
                    //	for (i = 0; i < p_row; ++i){
                    //		for (j = 0; j < p_col; ++j){

                    truth_t[thread_mod_id].resize(batch_size);
                    for (int k = 0; k < batch_size; k++)
                        truth_t[thread_mod_id][k] =  -FLT_MAX;
                    int out_po = ii;
                    int kernel_height = aux[0].kernel_size;
                    int kelnel_width = aux[1].kernel_size;
                    for (int k = 0; k < kernel_height; ++k){
                        for (int l = 0; l < kelnel_width; ++l){
                            //iii: current row 
                            int iii = i * aux[0].stride + k - aux[0].pad[0];
                            if (iii < 0 || iii >= dim[dim_size-2]) continue;
                            //v0: starting index in the current row
                            //v_end: ending index in the current row
                            //in_po: current input index
                            //out_po: current output index
                            //max: current max batch
                            //f: current used input index
                            int v0 = (t * dim[output_dim_size - 2] + iii) * dim[output_dim_size - 1];
                            int v_end = v0 + dim[output_dim_size - 1];
                            int in_po = v0 + (l > aux[1].pad[0]? l - aux[1].pad[0] : 0) + aux[1].stride * j;
                            if (in_po >= v_end) continue;
                            engine->decrypt(ciphertext_pool[in_po], plaintext[thread_mod_id]);
                            engine->decode(plaintext[thread_mod_id], test_t[thread_mod_id]);
                            for (int m = 0; m < batch_size; m++)
                                if ( test_t[thread_mod_id][m] > truth_t[thread_mod_id][m])
                                    truth_t[thread_mod_id][m] = test_t[thread_mod_id][m], gradients[layer][out_po][m] = in_po;
                        }
                    }
                    engine->encode(truth_t[thread_mod_id], plaintext[thread_mod_id]);
                    engine->encrypt(plaintext[thread_mod_id], ciphertext_pool2[out_po]); 
                    }
                }
                // 3. return the ciphertexts.
                network_io.send_ciphertext(op, layer, ciphertext_pool2.data(), output_dim, output_dim_size);
                delete []aux;
                break;
            }

            case remote_ops::OP_MAX2D_BP:{
                conv_conf_t *aux =  new conv_conf_t[2];
                network_io.recv_data(aux, 2*sizeof(conv_conf_t));
                //it seems my max2d don't support padding and overlapped input pixels for now.
                assert((aux[0].pad[0] == 0 && aux[0].pad[1] == 0 && aux[1].pad[0] == 0 && aux[1].pad[1] == 0));
                assert(aux[0].stride==aux[0].kernel_size && aux[1].stride==aux[1].kernel_size);
                assert(dim_size == 3);
                assert(dim[1]%aux[0].stride==0);
                assert(dim[2]%aux[1].stride==0);
                int output_number = 1, output_dim_size;
                int output_dim[5];
                //1.reversely get the previous layer size
                if (dim_size != 3) return -1;
                output_dim_size = 3;
                output_dim[0] = dim[0], output_dim[1] = conv_in_size(dim[1], &aux[0]), output_dim[2] = conv_in_size(dim[2], &aux[1]);
                for(int i=0; i< output_dim_size; i++)
                    output_number *= output_dim[i];


                while(ciphertext_pool2.size() < output_number){
                    SEALCiphertext tmp(engine);
                    ciphertext_pool2.push_back(tmp);
                }
                //2.standard backward process
                //do something like decrypt/encrypt
                int rest = 1, i, ii;
                int out_len = number;
                for (i = 0; i < dim_size - 2; ++i) rest *= dim[i];
                int p_row = dim[dim_size - 2], p_col = dim[dim_size - 1];
                assert(out_len == rest * p_row * p_col);
                #pragma omp parallel
                {
                #pragma omp for
                    for ( ii = 0; ii < out_len; ii++){
                        int t = ii / (p_row * p_col);
                        int i = (ii / p_col) % (p_row);
                        int j = (ii % p_col);
                        int thread_mod_id = omp_get_thread_num()%omp_get_max_threads();

                        truth_t[thread_mod_id].resize(batch_size);
                        test_t[thread_mod_id].resize(batch_size);
                        std::vector<double> & out_gradient = truth_t[thread_mod_id];
                        std::vector<double> & in_gradient = test_t[thread_mod_id];

                        int out_po = ii;
                        int kernel_height = aux[0].kernel_size;
                        int kelnel_width = aux[1].kernel_size;
                        engine->decrypt(ciphertext_pool[out_po], plaintext[thread_mod_id]);
                        engine->decode(plaintext[thread_mod_id], out_gradient);
                        for (int k = 0; k < kernel_height; ++k){
                            for (int l = 0; l < kelnel_width; ++l){
                                //iii: current row 
                                int iii = i * aux[0].stride + k - aux[0].pad[0];
                                if (iii < 0 || iii >= output_dim[dim_size-2]) continue;
                                //v0: starting index in the current row
                                //v_end: ending index in the current row
                                //in_po: current input index
                                //out_po: current output index
                                //max: current max batch
                                //f: current used input index
                                int v0 = (t * output_dim[dim_size - 2] + iii) * output_dim[dim_size - 1];
                                int v_end = v0 + output_dim[dim_size - 1];
                                int in_po = v0 + (l > aux[1].pad[0]? l - aux[1].pad[0] : 0) + aux[1].stride * j;
                                //cout << "inpo:" << in_po << "outpo:" << out_po << endl;
                                if (in_po >= v_end) continue;
                                // construct the gradient on the in_po position
                                for (int m = 0; m < batch_size; m++)
                                    if ( gradients[layer][out_po][m] == in_po )
                                        in_gradient[m] = out_gradient[m];
                                    else
                                        in_gradient[m] = 0;
                                engine->encode(in_gradient, plaintext[thread_mod_id]);
                                engine->encrypt(plaintext[thread_mod_id], ciphertext_pool2[in_po]); 
                            }
                        }
                    }
                }
                // 3. return the ciphertexts.
                network_io.send_ciphertext(op, layer, ciphertext_pool2.data(), output_dim, output_dim_size);
                delete []aux;
                break;
            }
        
            case remote_ops::OP_SOFTMAX_FP:{
                //1. recv extra data
                //   set the batch_size
                //   set the output number, dim, dim_size
                int output_number = number, output_dim_size = dim_size;
                int *output_dim = dim;
                //2.standard fp procedure, cache the related values in gradients[layer][dim_idx][batch_idx]
                int n1 = dim[dim_size - 1];
                vector<vector<double>> &raw_value = gradients[layer];
                vector<double> &y = test_t[0];
                vector<double> &max = truth_t[0]; 
                float s;
                y.resize(batch_size);
                max.resize(batch_size);
                for (j = 0; j < batch_size; ++j)
                    max[j] = -FLT_MAX;
                for (i = 0; i < n1; i++){
                    engine->decrypt(ciphertext_pool[i],*plaintext);
                    engine->decode(*plaintext,raw_value[i]);
                    for (j = 0; j < batch_size; ++j)
                        max[j] = max[j] > raw_value[i][j]? max[j] : raw_value[i][j];
                }
                for (j = 0; j < batch_size; ++j){
                    for (i = 0, s = 0.0f; i < n1; i++){
                        raw_value[i][j] = expf(raw_value[i][j]-max[j]);
                        s += raw_value[i][j];
                    }
                    for (i = 0, s = 1.0f /s; i < n1; ++i)
                        raw_value[i][j] *= s;
                }
                for (i = 0; i < n1; i++){
                    engine->encode(raw_value[i], *plaintext);
                    engine->encrypt(*plaintext, ciphertext_pool[i]);
                }
                // 3. return the ciphertexts.
                network_io.send_ciphertext(op, layer, ciphertext_pool.data(), output_dim, output_dim_size);
                
                break;
            }


            case remote_ops::OP_SOFTMAX_BP:{
                //1. recv extra data
                //   set the output number, dim, dim_size
                int output_number = number, output_dim_size = dim_size;
                int *output_dim = dim;
                //2.standard procedure
                float s;
                int n1 = dim[dim_size - 1];
                vector<vector<double>> &raw_value = gradients[layer];
                vector<vector<double>> raw_gradient(n1, vector<double>(batch_size, 0.));
                for (i = 0; i < n1; i++){
                    engine->decrypt(ciphertext_pool[i],*plaintext);
                    engine->decode(*plaintext,raw_gradient[i]);
                }
                // for xi, the gradient is g1*y1*(1-y1)-g2*y1*y2-g3*y1*y3 = yi(gi-(\Sigma_i y_i*g_i))
                for (j = 0; j < batch_size; ++j){
                    for (i = 0, s = 0.0f; i < n1; i++)
                        s += raw_gradient[i][j] * raw_value[i][j];
                    for (i = 0; i < n1; i ++)
                        raw_value[i][j] = raw_value[i][j]*(raw_gradient[i][j] - s);
                }
                for (i = 0; i < n1; i++){
                    engine->encode(raw_value[i], *plaintext);
                    engine->encrypt(*plaintext, ciphertext_pool[i]);
                } 

                // 3. return the ciphertexts.
                network_io.send_ciphertext(op, layer, ciphertext_pool.data(), output_dim, output_dim_size);
                break;

            }

/*
            case remote_ops::TEMPLATE:{
                //1. recv extra data
                //   set the output number, dim, dim_size
                int batch_size = ciphertext_pool[0].size();
                int output_number = number, output_dim_size = dim_size;
                int *output_dim = dim;
                break;
                //2.standard procedure , cache the related values in gradients[layer][dim_idx][batch_idx]

                //3.recv ciphertexts
                network_io.send_ciphertext(op, layer, ciphertext_pool.data(), output_number, output_dim, output_dim_size);
            }

*/
            case remote_ops::OP_DP_DECRYPTION:{
                int total_num = number;
                int current_id = 0;
                int handled_layer = 0;
                int current_number = number;
                vector<int> value_count_list;
                value_count_list.push_back(number);
                double global_norm[batch_size];
                for(int m = 0; m < batch_size; m ++)
                    global_norm[m] = 0.0;
                //maybe a series of recv_ciphertext
                while(true){
                    //1.handle the ciphertexts: decrypt them. 
                    //2.prepare the gradients vector
                    //3.put them into the gradients
                    while(gradients4dp.size() < total_num){
                        gradients4dp.push_back(vector<double>());
                    }
                    #pragma omp parallel
                    {
                    #pragma omp for
                        for(j = 0; j < current_number; ++j){
                            int thread_mod_id = omp_get_thread_num()%omp_get_max_threads();
                            if (ciphertext_pool[j].clean()) {
                                for( int m=0; m< batch_size; m++)
                                    gradients4dp[current_id+j][m] = 0.0;
                            }else{
                                engine->decrypt(ciphertext_pool[j], plaintext[thread_mod_id]);
                                engine->decode(plaintext[thread_mod_id], gradients4dp[current_id+j]);
                                #pragma omp critical
                                {
                                    for( int m = 0; m < batch_size; m++)
                                        global_norm[m] += pow(gradients4dp[current_id+j][m], 2);
                                }
                            }
                        }
                    }
                    current_id += current_number;
                    assert(current_id == total_num);
                    handled_layer++;
                    if(handled_layer == layer)
                        break;
                    current_number = network_io.recv_ciphertext(ciphertext_pool);
                    cout << "more decrypted ciphertext number:" << current_number << endl;
                    value_count_list.push_back(current_number);
                    total_num += current_number;
                }
                assert(layer == value_count_list.size());
                //norm for every batch
                for (int m = 0; m < batch_size; m ++){
                    global_norm[m] = sqrt(global_norm[m]);
                    global_norm[m] = max(1.0, global_norm[m]/l2_norm_clip);
                }

                int starting_idx = 0;
                int ending_idx;
                double sum, noise_sum;
                if(gradients4dp.size() < total_num){
                    gradients4dp.push_back(vector<double>(batch_size, 0.0));
                }
                //total_num number of batch_size vector, following one more (total_num) vector. 
                processed_gradients.resize(total_num);
                for(j = 0; j< layer; j++){
                    ending_idx = starting_idx + value_count_list[j];
                    for( int i = starting_idx; i< ending_idx; i++){
                        //norm
                        for(int m = 0; m < batch_size; m++)
                            gradients4dp[i][m] /= global_norm[m];
                        //sum
                        //cout << gradients4dp[i][0] <<gradients4dp[i][1] << gradients4dp[i].size() << endl;
                        sum = std::accumulate(gradients4dp[i].begin(), gradients4dp[i].end(), 0.0);
                        //add noise
                        noise_sum = sum;// + dist(random_generator);
                        //multiply learning rate
                        //cout << "sum" << sum << endl;
                        processed_gradients[i] = noise_sum;
                    }
                    network_io.send_data(&(processed_gradients.data()[starting_idx]), (value_count_list[j])* sizeof(float));
                    starting_idx = ending_idx;
                }
                break;
            }

            default:{
                cout << "unknown op!" << endl;
                flag = false;
                break;
            }
        }
        //for(int i=0;i< number ;i++)
        //    print_ciphertext(&ciphertext_pool[i]);

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
