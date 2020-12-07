
#include <cstdlib>
#include <unistd.h>
#include <cassert>
#include <cstdio>
#include <chrono>
#include <iomanip>
#include <iostream>
#include "kann.h"
#include "util.h"
#include "kann_extra/kann_data.h"
#include <omp.h>

using namespace std;

static kann_t *lenet_gen(unsigned int n_labels)
{
    kad_node_t *lenet;

    assert(n_labels > 0);

    lenet = kad_feed(3, 1, 28, 28), lenet->ext_flag |= KANN_F_IN;   //because we don't have batch, thus the dimension num is 3.
    lenet = kann_layer_conv2d(lenet, 6, 5, 5, 1, 1, 1, 1);
    lenet = kad_max2d(kad_relu(lenet), 2, 2, 2, 2, 0, 0); // 2x2 kernel; 0x0 stride; 0x0 padding
    lenet = kann_layer_conv2d(lenet, 16, 5, 5, 1, 1, 0, 0);
    lenet = kad_max2d(kad_relu(lenet), 2, 2, 2, 2, 0, 0);
    lenet = kad_relu(kann_layer_dense(lenet, 120));
    lenet = kad_relu(kann_layer_dense(lenet, 84));

    if (n_labels == 1)
        return kann_new(kann_layer_cost(lenet, n_labels, KANN_C_CEB), 0);
    else
        return kann_new(kann_layer_cost(lenet, n_labels, KANN_C_CEM), 0);
}

int main(int argc, char *argv[])
{
    //0. set the environment
    int max_epoch = 50, mini_size = 64, max_drop_streak = 10, loss_type = KANN_C_CEB;
    int i, j, k, c, seed = 11, n_threads = 1, lazy_mode = 1;
    int total_samples;
    int mode = 0;
    kann_data_t *in = 0;
    kann_data_t *out = 0;
    kann_t *ann = 0;
    char *out_fn = 0, *in_fn = 0;
    float lr = 0.001f, frac_val = 0.1f;
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds inference_time(0);
    chrono::microseconds training_step_time(0);
    chrono::microseconds training_time(0);
    cout << "start reading parameters" << endl;
    cout << "argc:" << argc << endl;
    while ((c = getopt(argc, argv, "s:r:m:B:v:z:M:")) >= 0) {
        if (c == 's') seed = atoi(optarg);
        //else if (c == 'i') in_fn = optarg;
        //else if (c == 'o') out_fn = optarg;
        else if (c == 'r') lr = atof(optarg);
        else if (c == 'm') max_epoch = atoi(optarg);
        else if (c == 'B') mini_size = atoi(optarg);
        else if (c == 'v') frac_val = atof(optarg);
        else if (c == 'z') lazy_mode = atoi(optarg);
        else if (c == 'M') mode = atoi(optarg);
        //else if (c == 't') n_threads = atoi(optarg);
    }
    //argc - optind < 1, which means no more other arguments followed, i.e., training feature/label files missing.
    if (argc - optind < 1) {
        FILE *fp = stdout;
        fprintf(fp, "Usage: mlp [options] <in.knd> [truth.knd]\n");
        fprintf(fp, "Options:\n");
        fprintf(fp, "  Model construction:\n");
        //fprintf(fp, "    -i FILE     read trained model from FILE []\n");
        //fprintf(fp, "    -o FILE     save trained model to FILE []\n");
        fprintf(fp, "    -s INT      random seed [%d]\n", seed);
        fprintf(fp, "  Model training:\n");
        fprintf(fp, "    -M INT     working mode: 0-all optimized 1-zero_encryption 2-lazy_relinearize 3-no optimization [%g]\n", mode);
        fprintf(fp, "    -r FLOAT    learning rate [%g]\n", lr);
        fprintf(fp, "    -m INT      max number of epochs [%d]\n", max_epoch);
        fprintf(fp, "    -B INT      mini-batch size [%d]\n", mini_size);
        fprintf(fp, "    -z int      lazy mode [%d]\n", lazy_mode);
        fprintf(fp, "    -v FLOAT    fraction of data used for validation [%g]\n", frac_val);
        //fprintf(fp, "    -t INT      number of threads [%d]\n", n_threads);
        return 1;
    }

    // the encryption environment
    cout << "set up the encryption engine" << endl;
    size_t poly_modulus_degree = 8192;
    size_t standard_scale = 30;
    std::vector<int> coeff_modulus = {40, 30, 30, 30, 30, 40};
    SEALEncryptionParameters parms(poly_modulus_degree,
                    coeff_modulus,
                    seal_scheme::CKKS);
    engine = make_shared<SEALEngine>();
    engine->init(parms, standard_scale, false);
	engine->max_slot() = mini_size;
    size_t slot_count = engine->slot_count();
    cout <<"Poly modulus degree: " << poly_modulus_degree<< endl;
    cout << "Coefficient modulus: ";
    cout << endl;
    cout << "slot count: " << slot_count<< endl;
    cout << "scale: 2^" << standard_scale << endl;

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
		t->resize(engine->max_slot());
	}
	test_t = new vector<double>[max_parallel];
	for (i = 0; i < max_parallel; i ++){
		test_t->resize(engine->max_slot());
	}
	truth_t = new vector<double>[max_parallel];
	for (i = 0; i < max_parallel; i ++){
		truth_t->resize(engine->max_slot());
	}
    if(mode == 1 || mode ==0){
        engine->zero = new SEALCiphertext(engine);
        engine->encode(0, *plaintext);
        engine->encrypt(*plaintext, *(engine->zero));
    }else{
        engine->zero = NULL;
    }
    if(mode == 2 || mode == 0) engine->lazy_relinearization() = true;
    else engine->lazy_relinearization() = false;
    engine->lazy_mode() = lazy_mode;
    //1. read the model and data
    kann_srand(seed);
    in = kann_data_read(argv[optind]);
    total_samples = 128;
    //total_samples = in->n_row;
    if (in_fn) {
        ann = kann_load(in_fn);
        assert(kann_dim_in(ann) == in->n_col);
    }

    if (optind + 1 < argc) { // read labels
        out = kann_data_read(argv[optind + 1]);
        cout << "in row: " << in->n_row << "\tout row: " << out->n_row << endl;
        assert(in->n_row == out->n_row);
    }
    //for now only training 
    if (!out){
        fprintf(stdout, "For now only training is implemented, labels needed.");
        return -1;
    }

    printf("in col: %d, out col: %d\n", in->n_col, out->n_col);
    assert(in->n_col == 1*28*28);
    assert(out->n_col == 10);

    // shuffling, batching and encrypting the data. (packaging into a cpp file later)
    
    SEALPlaintext tmp(engine);
    int * shuf = (int*)malloc(total_samples * sizeof(int));
    for (i = 0; i < total_samples; ++i) shuf[i] = i;
    if (out){
        // only need to shuffle when labels are provided for training.
        kann_shuffle(total_samples, shuf);  
    }
    
    double train_cost;

    kann_data_t *used_dataset = in;
    kann_data_t *used_label = out;

    // 5.5 test the cnn
    //create a pseudo image dataset
    cout << "Starting testing cnn." << endl;

    //batching: in secure training we can use different batch size for every batch. 
    int left_sample_num = total_samples;
    int current_sample_id = 0;
    int current_cipher_id = 0;
    int cipher_num = total_samples % mini_size == 0? total_samples / mini_size : total_samples / mini_size + 1;
    vector<vector<SEALCiphertext>> image_features_c(cipher_num, vector<SEALCiphertext>(used_dataset->n_col, SEALCiphertext(engine)));
    vector<vector<SEALCiphertext>> image_labels_c(cipher_num, vector<SEALCiphertext>(used_label->n_col, SEALCiphertext(engine)));
    vector<vector<double>> image_features;
    vector<vector<double>> image_labels;
    image_features.resize(used_dataset->n_col);
    image_labels.resize(used_label->n_col);

    while (left_sample_num > 0){
        int batch_size = left_sample_num >= mini_size? mini_size : left_sample_num;
        for (j = 0; j < used_dataset->n_col; j++)
            image_features[j].resize(batch_size);
        for (j = 0; j < used_label->n_col; j++)
            image_labels[j].resize(batch_size);
        for (k = current_sample_id; k < current_sample_id + batch_size; k++){
            for (j = 0; j < used_dataset->n_col; j++){
                image_features[j][k] = used_dataset->x[shuf[k]][j];         
            }
            for (j = 0; j < used_label->n_col; j++){
                image_labels[j][k] = used_label->x[shuf[k]][j];         
            }
        }
        for (j = 0; j < used_dataset->n_col; j++){
            engine->encode(image_features[j], tmp);
            engine->encrypt(tmp, image_features_c[current_cipher_id][j]);
        }
        for (j = 0; j < used_label->n_col; j++){
            engine->encode(image_labels[j], tmp);
            engine->encrypt(tmp, image_labels_c[current_cipher_id][j]);
        }
        current_cipher_id ++;
        current_sample_id += batch_size;
        left_sample_num -= batch_size;
    }
    assert(current_cipher_id == cipher_num);
    fprintf(stdout, "Finishing image data encryption.\n");
    cout << "sample num:"<< current_sample_id << endl;
    cout << "cipher batch num:"<< current_cipher_id << endl;

    // gen lenet
    kann_t *cnn_ann = lenet_gen(used_label->n_col);

    cout << "CNN Back propagation start" << endl;
    kann_switch(cnn_ann, 1);

    SEALCiphertext * bind_data = image_features_c[0].data();
    SEALCiphertext * bind_label = image_labels_c[0].data();  
    kann_feed_bind(cnn_ann, KANN_F_IN, 0, &bind_data);
    kann_feed_bind(cnn_ann, KANN_F_TRUTH, 0, &bind_label);
    train_cost = kann_cost(cnn_ann, 0, 1);

    for (k = 0; k < cnn_ann->n; k++){
        if (kad_is_var(cnn_ann->v[k])){
            if(seal_is_encrypted(cnn_ann->v[k]))
                kann_SGD(kad_len(cnn_ann->v[k]), lr, 0, cnn_ann->v[k]->g_c, cnn_ann->v[k]->x_c);
            else
                kann_SGD(kad_len(cnn_ann->v[k]), lr, 0, cnn_ann->v[k]->g, cnn_ann->v[k]->x);    
         }
    }
    cout <<"switch to eval" <<endl;
    time_start = chrono::high_resolution_clock::now();
    kann_switch(cnn_ann, 0);
    train_cost = kann_cost(cnn_ann, 0, 0);
    time_end = chrono::high_resolution_clock::now();
    //int i_cost = kann_find(cnn_ann, KANN_F_COST, 0);
    training_step_time = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    //print_model(cnn_ann, i_cost, 1);
    cout << "CNN Training step time:" << training_step_time.count() << "microseconds" << endl;

    cout << "eval class error" << endl;
    int b;
    vector<float> truth(mini_size * used_label->n_col, 0);
    c = kann_class_error(cnn_ann, truth.data(), &b);
    cout << "error num: " << c << endl;

    // 6. finish the training 
    
    //kann_train_fnn1(ann, lr, mini_size, max_epoch, max_drop_streak, frac_val, in->n_row, in->x, out->x);



    //delete engine->zero;
	delete shuf;
	delete[] plaintext;
	delete[] ciphertext;
	delete[] t;
	delete[] truth_t;
	delete[] test_t;
	kann_data_free(out);
	kann_data_free(in);
	kann_delete(ann);
	kann_delete(cnn_ann);
	return 0;
}
