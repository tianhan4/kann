
#include <cstdlib>
#include <unistd.h>
#include <cassert>
#include <cstdio>
#include <chrono>
#include <iomanip>
#include <iostream>
#include "kann.h"
#include "kann_extra/kann_data.h"

using namespace std;


static kann_t *model_gen(int n_in, int n_out, int loss_type, int n_h_layers, int n_h_neurons, float h_dropout, bool w_is_encrypted, bool b_is_encrypted)
{
    int i;
    kad_node_t *t;
    t = kann_layer_input(n_in);
    for (i = 0; i < n_h_layers; ++i)
        t = kad_relu(kann_layer_dense(t, n_h_neurons, w_is_encrypted, b_is_encrypted));
        //t = kann_layer_dropout(kad_relu(kann_layer_dense(t, n_h_neurons, w_is_encrypted, b_is_encrypted)), h_dropout);
         //better put the last layer all encrypted. For we can infer the i-1 activations from the gradients of w, due to the situation where only 1 node output.
    return kann_new(kann_layer_cost(t, n_out, loss_type, true, true), 0);
}

static kann_t *lenet_gen(unsigned int n_labels)
{
    kad_node_t *lenet;

    assert(n_labels > 0);

    lenet = kad_feed(3, 1, 28, 28), lenet->ext_flag |= KANN_F_IN;   //because we don't have batch, thus the dimension num is 3.
    lenet = kann_layer_conv2d(lenet, 6, 3, 3, 1, 1, 1, 1);
    lenet = kad_max2d(kad_relu(lenet), 2, 2, 2, 2, 0, 0); // 2x2 kernel; 0x0 stride; 0x0 padding
    lenet = kann_layer_conv2d(lenet, 16, 3, 3, 1, 1, 0, 0);
    lenet = kad_max2d(kad_relu(lenet), 2, 2, 2, 2, 0, 0);
    lenet = kad_relu(kann_layer_dense(lenet, 120));
    lenet = kad_relu(kann_layer_dense(lenet, 84));
    lenet = kad_relu(kann_layer_dense(lenet, n_labels));

    return kann_new(kann_layer_cost(lenet, n_labels, KANN_C_CEB), 0);
}


template<typename T>
static inline void print_vector(std::vector<T> vec, size_t print_size = 4, int prec = 3)
{
    /*
    Save the formatting information for std::cout.
    */
    std::ios old_fmt(nullptr);
    old_fmt.copyfmt(std::cout);

    size_t slot_count = vec.size();

    std::cout << std::fixed << std::setprecision(prec);
    std::cout << std::endl;
    if(slot_count <= 2 * print_size)
    {
        std::cout << "    [";
        for (size_t i = 0; i < slot_count; i++)
        {
            std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
        }
    }
    else
    {
        vec.resize(std::max(vec.size(), 2 * print_size));
        std::cout << "    [";
        for (size_t i = 0; i < print_size; i++)
        {
            std::cout << " " << vec[i] << ",";
        }
        if(vec.size() > 2 * print_size)
        {
            std::cout << " ...,";
        }
        for (size_t i = slot_count - print_size; i < slot_count; i++)
        {
            std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
        }
    }
    //std::cout << "size: " << vec.size();
    std::cout << std::endl;

    /*
    Restore the old std::cout formatting.
    */
    std::cout.copyfmt(old_fmt);
}

void print_ciphertext(SEALCiphertext *cipher){
    engine->decrypt(*cipher, *plaintext);
    engine->decode(*plaintext, t);
    cout << t.size() << endl;
    print_vector(t);
}

void print_model(kann_t * model, int from, bool grad){
    int i,j,k;
    assert(from < model->n);
    cout << "total node num:" << model->n << endl;
    for(i = 0; i <= from ;i++){
        if (kad_is_feed(model->v[i])){
            cout << "node " << i << ": " << "feed" << " size: " << kad_len(model->v[i]) << endl;
        }
        else if (kad_is_var(model->v[i])) {
            cout << "node " << i << ": " << "leaf" << " size: " << kad_len(model->v[i]) << endl;
        }
        else {
            cout << "node " << i << ": " << kad_op_name[model->v[i]->op] << " size: " << kad_len(model->v[i]) << endl;
        }
        if (kad_is_feed(model->v[i]) && model->v[i]->x_c){
            cout << "encrypted feed:" << endl;
            cout << " level: " << engine->get_context()->get_sealcontext()->get_context_data(model->v[i]->x_c[0].ciphertext().parms_id())->chain_index() << endl;
            cout << "ciphertext size:" << model->v[i]->x_c[0].size() << endl;
            for (j = 0; j < kad_len(model->v[i]); j++){
                engine->decrypt(model->v[i]->x_c[j], *plaintext);
                engine->decode(*plaintext, t);
                print_vector(t);
            }
        }
        else if (kad_is_back(model->v[i])){
            if(seal_is_encrypted(model->v[i])){
                cout << "encrypted:" << endl;
                cout << " level: " << engine->get_context()->get_sealcontext()->get_context_data(model->v[i]->x_c[0].ciphertext().parms_id())->chain_index() << endl;
                cout << "ciphertext size:" << model->v[i]->x_c[0].size() << endl;
                for (j = 0; j < kad_len(model->v[i]); j++){
                    engine->decrypt(model->v[i]->x_c[j], *plaintext);
                    engine->decode(*plaintext, t);
                    print_vector(t);
                }
                if(grad){
                    cout << "encrytped grad:" << endl;
                    if(model->v[i]->g_c[0].clean()){
                        cout << "clean grad" << endl;
                    }else{
                        for (j = 0; j < kad_len(model->v[i]); j++){
                            engine->decrypt(model->v[i]->g_c[j], *plaintext);
                            engine->decode(*plaintext, t);
                            print_vector(t);
                        }
                    }
                }
            }else{
                cout << "plain var:" << endl;
                for (j = 0; j < kad_len(model->v[i]); j++){
                    cout << model->v[i]->x[j] << endl;
                }
                if (grad){
                    cout << "plain grad:" << endl;
                    if(model->v[i]->g_c[0].clean()){
                        cout << "clean grad" << endl;
                    }else{
                        for (j = 0; j < kad_len(model->v[i]); j++){
                            cout << model->v[i]->g[j] << endl;
                        }
                    }
                }
            }
        }
    }
}


int main(int argc, char *argv[])
{
    //0. set the environment
    int max_epoch = 50, mini_size = 64, max_drop_streak = 10, loss_type = KANN_C_CEB;
    int i, j, k, c, n_h_neurons = 64, n_h_layers = 1, seed = 11, n_threads = 1;
    int total_samples;
    kann_data_t *in = 0;
    kann_data_t *out = 0;
    kann_t *ann = 0;
    char *out_fn = 0, *in_fn = 0;
    float lr = 0.001f, frac_val = 0.1f, h_dropout = 0.0f;
    chrono::high_resolution_clock::time_point time_start, time_end;
    chrono::microseconds inference_time(0);
    chrono::microseconds training_step_time(0);
    chrono::microseconds training_time(0);
    cout << "start reading parameters" << endl;
    cout << "argc:" << argc << endl;
    while ((c = getopt(argc, argv, "n:l:s:r:m:B:d:v:M")) >= 0) {
        if (c == 'n') n_h_neurons = atoi(optarg);
        else if (c == 'l') n_h_layers = atoi(optarg);
        else if (c == 's') seed = atoi(optarg);
        //else if (c == 'i') in_fn = optarg;
        //else if (c == 'o') out_fn = optarg;
        else if (c == 'r') lr = atof(optarg);
        else if (c == 'm') max_epoch = atoi(optarg);
        else if (c == 'B') mini_size = atoi(optarg);
        else if (c == 'd') h_dropout = atof(optarg);
        else if (c == 'v') frac_val = atof(optarg);
        else if (c == 'M') loss_type = KANN_C_CEM;
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
        fprintf(fp, "    -l INT      number of hidden layers [%d]\n", n_h_layers);
        fprintf(fp, "    -n INT      number of hidden neurons per layer [%d]\n", n_h_neurons);
        fprintf(fp, "    -d FLOAT    dropout at the hidden layer(s) [%g]\n", h_dropout);
        fprintf(fp, "    -M          use multi-class cross-entropy (binary by default)\n");
        fprintf(fp, "  Model training:\n");
        fprintf(fp, "    -r FLOAT    learning rate [%g]\n", lr);
        fprintf(fp, "    -m INT      max number of epochs [%d]\n", max_epoch);
        fprintf(fp, "    -B INT      mini-batch size [%d]\n", mini_size);
        fprintf(fp, "    -v FLOAT    fraction of data used for validation [%g]\n", frac_val);
        //fprintf(fp, "    -t INT      number of threads [%d]\n", n_threads);
        return 1;
    }

    // the encryption environment
    cout << "set up the encryption engine" << endl;
    size_t poly_modulus_degree = 8192;
    size_t standard_scale = 40;
    std::vector<int> coeff_modulus = {60, 40, 40, 60};
    SEALEncryptionParameters parms(poly_modulus_degree,
                    coeff_modulus,
                    seal_scheme::CKKS);
    engine = make_shared<SEALEngine>();
    engine->init(parms, standard_scale, false);
    engine->max_slot() = 64;
    size_t slot_count = engine->slot_count();
    cout <<"Poly modulus degree: " << poly_modulus_degree<< endl;
    cout << "Coefficient modulus: ";
    cout << endl;
    cout << "slot count: " << slot_count<< endl;
    cout << "scale: 2^" << standard_scale << endl;
    plaintext = new SEALPlaintext(engine);
    ciphertext = new SEALCiphertext(engine);
    engine->zero = new SEALCiphertext(engine);
    engine->encode(0, *plaintext);
    engine->encrypt(*plaintext, *(engine->zero));
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
    time_start = chrono::high_resolution_clock::now();
    train_cost = kann_cost(cnn_ann, 0, 1);
    time_end = chrono::high_resolution_clock::now();
    int i_cost = kann_find(cnn_ann, KANN_F_COST, 0);
    training_step_time = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    print_model(cnn_ann, i_cost, 1);
    cout << "CNN Training step time:" << training_step_time.count() << "microseconds" << endl;

    // 5.6 check cnn gradients
    kad_check_grad(cnn_ann->n, cnn_ann->v, i_cost);


    // 6. finish the training 
    
    //kann_train_fnn1(ann, lr, mini_size, max_epoch, max_drop_streak, frac_val, in->n_row, in->x, out->x);

    kann_data_free(in);
    kann_data_free(out);
    delete shuf;
    delete plaintext;
    delete ciphertext;
    kann_data_free(out);
    kann_data_free(in);
    kann_delete(ann);
    kann_delete(cnn_ann);
    return 0;
}
