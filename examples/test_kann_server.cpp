#include <cstdlib>
#include <cmath>
#include <iostream>
#include <omp.h>
#include "NetIO.h"
#include "kann.h"
#include "kann_extra/kann_data.h"
#include "util.h"

using namespace std;

static kann_t *lenet_gen(unsigned int n_labels)
{
    kad_node_t *lenet;

    assert(n_labels > 0);

    lenet = kad_feed(3, 1, 28, 28), lenet->ext_flag |= KANN_F_IN;   //because we don't have batch, thus the dimension num is 3.
    lenet = kann_layer_conv2d(lenet, 5, 5, 5, 2, 2, KAD_PAD_SAME, KAD_PAD_SAME, false);
    lenet = kann_layer_bias(lenet, false);
    lenet = kad_relu(lenet);
    lenet = kad_relu(kann_layer_dense(lenet, 100, false, false));
    
    ///** best model for mnist
    //lenet = kann_layer_conv2d(lenet, 16, 8, 8, 2, 2, KAD_PAD_SAME, KAD_PAD_SAME, false);
    //lenet = kann_layer_bias(lenet, true);
    //lenet = kad_max2d(kad_relu(lenet), 2, 2, 1, 1, 0, 0); // 2x2 kernel; 1x1 stride; 0x0 padding
    //lenet = kann_layer_conv2d(lenet, 32, 4, 4, 2, 2, 0, 0,false);
    //lenet = kann_layer_bias(lenet, true);
    //lenet = kad_max2d(kad_relu(lenet), 2, 2, 1, 1, 0, 0); // 2x2 kernel; 1x1 stride; 0x0 padding
    //lenet = kad_relu(kann_layer_dense(lenet, 32,false));
    //**/

    // lenet = kann_layer_conv2d(lenet, 16, 5, 5, 1, 1, 0, 0);
    //lenet = kann_layer_conv2d(lenet, 3, 5, 5, 1, 1, 0, 0);
    //lenet = kad_max2d(kad_relu(lenet), 2, 2, 2, 2, 0, 0);
    // lenet = kad_relu(kann_layer_dense(lenet, 120));
    // lenet = kad_relu(kann_layer_dense(lenet, 84));

    if (n_labels == 1)
        return kann_new(kann_layer_cost(lenet, n_labels, KANN_C_CEB, false, false), 0);
    else
        return kann_new(kann_layer_cost(lenet, n_labels, KANN_C_CEM, false, false), 0);
}

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

    cout << "wait for someone to connect." << endl;
    NetIO network_io(nullptr, 17722, engine);
    cout << "connected" << endl;
    int layer = 0;
    int ciphertext_num, gradient_num, recv_num;
    SEALCiphertext small_ciphertext[100];
    SEALCiphertext small_ciphertext2[100];
    float returned_noisy[100];
    int small_batch_size;
    int dim[5];
    int dim_size;
    vector<vector<double>> raw_values;
    vector<vector<double>> raw_labels;
    vector<vector<double>> raw_gradients;
    //=======
    //TEST RELU
    ciphertext_num=4;
    small_batch_size = 2;
    dim[0] = ciphertext_num;
    dim_size = 1;
    raw_values.clear();
    raw_values.push_back(vector<double>{-1,2});
    raw_values.push_back(vector<double>{2,-1});
    raw_values.push_back(vector<double>{-1,-1});
    raw_values.push_back(vector<double>{2,2});
    for( int i = 0; i< ciphertext_num;i++){
        engine->encode(raw_values[i], *plaintext);
        engine->encrypt(*plaintext, small_ciphertext[i]);
    }

    network_io.send_ciphertext(remote_ops::OP_RELU_FP, layer, small_ciphertext, dim, dim_size);
    recv_num = network_io.recv_ciphertext(small_ciphertext);
    cout << "RELU FP Result:" << endl;
    for(int i=0;i<recv_num;i++)
        print_ciphertext(&small_ciphertext[i]);

    gradient_num = ciphertext_num;
    raw_gradients.clear();
    raw_gradients.push_back(vector<double>{1,4});
    raw_gradients.push_back(vector<double>{2,3});
    raw_gradients.push_back(vector<double>{3,2});
    raw_gradients.push_back(vector<double>{4,1});
    for( int i = 0; i< gradient_num;i++){
        engine->encode(raw_gradients[i], *plaintext);
        engine->encrypt(*plaintext, small_ciphertext[i]);
    }
    network_io.send_ciphertext(remote_ops::OP_RELU_BP, layer, small_ciphertext, dim, dim_size);
    recv_num = network_io.recv_ciphertext(small_ciphertext);
    cout << "RELU BP Result:" << endl;
    for(int i=0;i<recv_num;i++)
        print_ciphertext(&small_ciphertext[i]);

    layer++;
    //===
    //TEST CE_MULTI
    ciphertext_num=4;
    small_batch_size = 2;
    dim[0] = ciphertext_num;
    dim_size = 1;
    raw_values.clear();
    raw_labels.clear();
    raw_values.push_back(vector<double>{0.5, 0.1});
    raw_values.push_back(vector<double>{0, 0.2});
    raw_values.push_back(vector<double>{0.5, 0.3});
    raw_values.push_back(vector<double>{0, 0.4});
    raw_labels.push_back(vector<double>{0,0});
    raw_labels.push_back(vector<double>{0,0});
    raw_labels.push_back(vector<double>{1,0});
    raw_labels.push_back(vector<double>{0,1});
    for( int i = 0; i< ciphertext_num;i++){
        engine->encode(raw_values[i], *plaintext);
        engine->encrypt(*plaintext, small_ciphertext[i]);
    }
    for( int i = 0; i< ciphertext_num;i++){
        engine->encode(raw_labels[i], *plaintext);
        engine->encrypt(*plaintext, small_ciphertext2[i]);
    }

    network_io.send_ciphertext(remote_ops::OP_CE_MULTI_FP, layer, small_ciphertext, dim, dim_size);
    network_io.send_ciphertext(remote_ops::OP_CE_MULTI_FP, layer, small_ciphertext2, dim, dim_size);
    recv_num = network_io.recv_ciphertext(small_ciphertext);
    assert(recv_num == 1);
    cout << "OP_CE_MULTI_FP:" << endl;\
        print_ciphertext(&small_ciphertext[0]);

    
    gradient_num = 1;
    raw_gradients.clear();
    raw_gradients.push_back(vector<double>{1,1});
    for( int i = 0; i< gradient_num;i++){
        engine->encode(raw_gradients[i], *plaintext);
        engine->encrypt(*plaintext, small_ciphertext[i]);
    }
    dim[0] = 1;
    network_io.send_ciphertext(remote_ops::OP_CE_MULTI_BP, layer, small_ciphertext, dim, 1);
    recv_num = network_io.recv_ciphertext(small_ciphertext);
    cout << "OP_CE_MULTI_BP:" << endl;
    for(int i=0;i<recv_num;i++)
        print_ciphertext(&small_ciphertext[i]);

    layer++;
    //===
    //TEST MAX2D

    ciphertext_num=16;
    small_batch_size = 2;
    dim[0] = 1; dim[1] = 4; dim[2] = 4;
    dim_size = 3;
    raw_values.clear();
    raw_values.push_back(vector<double>{1,1});
    raw_values.push_back(vector<double>{2,4});
    raw_values.push_back(vector<double>{6,9});
    raw_values.push_back(vector<double>{5,5});
    raw_values.push_back(vector<double>{3,3});
    raw_values.push_back(vector<double>{4,2});
    raw_values.push_back(vector<double>{8,7});
    raw_values.push_back(vector<double>{7,6});
    raw_values.push_back(vector<double>{4,1});
    raw_values.push_back(vector<double>{2,4});
    raw_values.push_back(vector<double>{7,7});
    raw_values.push_back(vector<double>{6,9});
    raw_values.push_back(vector<double>{3,3});
    raw_values.push_back(vector<double>{1,2});
    raw_values.push_back(vector<double>{5,6});
    raw_values.push_back(vector<double>{8,5});
    for( int i = 0; i< ciphertext_num;i++){
        engine->encode(raw_values[i], *plaintext);
        engine->encrypt(*plaintext, small_ciphertext[i]);
    }

    network_io.send_ciphertext(remote_ops::OP_MAX2D_FP, layer, small_ciphertext, dim, dim_size);

	conv_conf_t *aux = new conv_conf_t[2]; 
    aux[0].pad[0]=0;aux[0].pad[1]=0;aux[1].pad[0]=0;aux[1].pad[1]=0;
    aux[0].kernel_size=2;aux[1].kernel_size=2;
    aux[0].stride=2;aux[1].stride=2;
    network_io.send_data(aux, 2*sizeof(conv_conf_t));
    recv_num = network_io.recv_ciphertext(small_ciphertext);
    cout << "MAX2D FP Result:" << recv_num << endl;
    for(int i=0;i<recv_num;i++)
        print_ciphertext(&small_ciphertext[i]);

    
    gradient_num = 4;
    dim[0] = 1;dim[1] = 2;dim[2] = 2;
    dim_size = 3;
    raw_gradients.clear();
    raw_gradients.push_back(vector<double>{1,4});
    raw_gradients.push_back(vector<double>{2,3});
    raw_gradients.push_back(vector<double>{3,2});
    raw_gradients.push_back(vector<double>{4,1});
    for( int i = 0; i< gradient_num;i++){
        engine->encode(raw_gradients[i], *plaintext);
        engine->encrypt(*plaintext, small_ciphertext[i]);
    }
    network_io.send_ciphertext(remote_ops::OP_MAX2D_BP, layer, small_ciphertext, dim, dim_size);
    network_io.send_data(aux, 2*sizeof(conv_conf_t));
    recv_num = network_io.recv_ciphertext(small_ciphertext);
    cout << "MAX2D BP Result:" << endl;
    for(int i=0; i < recv_num; i++)
        print_ciphertext(&small_ciphertext[i]);
    delete []aux;

    layer++;
    //===
    //TEST SOFTMAX
    ciphertext_num=4;
    small_batch_size = 2;
    dim[0] = ciphertext_num;
    dim_size = 1;
    raw_values.clear();
    raw_values.push_back(vector<double>{log(1),log(4)});
    raw_values.push_back(vector<double>{log(2),log(3)});
    raw_values.push_back(vector<double>{log(3),log(2)});
    raw_values.push_back(vector<double>{log(4),log(1)});
    for( int i = 0; i< ciphertext_num;i++){
        engine->encode(raw_values[i], *plaintext);
        engine->encrypt(*plaintext, small_ciphertext[i]);
    }

    network_io.send_ciphertext(remote_ops::OP_SOFTMAX_FP, layer, small_ciphertext, dim, dim_size);
    recv_num = network_io.recv_ciphertext(small_ciphertext);
    cout << "SOFTMAX FP Result:" << endl;
    for(int i=0;i<recv_num;i++)
        print_ciphertext(&small_ciphertext[i]);

    gradient_num = ciphertext_num;
    raw_gradients.clear();
    raw_gradients.push_back(vector<double>{1,1});
    raw_gradients.push_back(vector<double>{2,2});
    raw_gradients.push_back(vector<double>{3,3});
    raw_gradients.push_back(vector<double>{4,4});
    for( int i = 0; i< gradient_num;i++){
        engine->encode(raw_gradients[i], *plaintext);
        engine->encrypt(*plaintext, small_ciphertext[i]);
    }
    network_io.send_ciphertext(remote_ops::OP_SOFTMAX_BP, layer, small_ciphertext, dim, dim_size);
    recv_num = network_io.recv_ciphertext(small_ciphertext);
    cout << "SOFTMAX BP Result:" << endl;
    for(int i=0;i<recv_num;i++)
        print_ciphertext(&small_ciphertext[i]);

    layer++;
    //===
    //TEST DP_DECRYPTION

    ciphertext_num=2;
    small_batch_size = 2;
    dim[0] = ciphertext_num;
    dim_size = 1;
    raw_values.clear();
    raw_values.push_back(vector<double>{1,0.2});
    raw_values.push_back(vector<double>{2,0.1});
    for( int i = 0; i< ciphertext_num;i++){
        engine->encode(raw_values[i], *plaintext);
        engine->encrypt(*plaintext, small_ciphertext[i]);
    }
    network_io.send_ciphertext(remote_ops::OP_DP_DECRYPTION, 2, small_ciphertext, dim, dim_size);

    raw_values.clear();
    raw_values.push_back(vector<double>{3,0.3});
    raw_values.push_back(vector<double>{4,0.1});
    for( int i = 0; i< ciphertext_num;i++){
        engine->encode(raw_values[i], *plaintext);
        engine->encrypt(*plaintext, small_ciphertext[i]);
    }
    network_io.send_ciphertext(remote_ops::OP_DP_DECRYPTION, 2, small_ciphertext, dim, dim_size);
    
    cout << "DP DECRYPTION Result:" << endl;
    for(int i = 0; i < 2; i++){
        network_io.recv_data(returned_noisy, 2*sizeof(float));
        for(int j=0; j < 2; j++)
            cout << returned_noisy[j] << " ";
        cout <<endl;
    }

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


