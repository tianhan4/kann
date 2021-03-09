#include <cstdlib>
#include <cmath>
#include <iostream>
#include "NetIO.h"
#include "kann.h"
#include "kann_extra/kann_data.h"
#include "util.h"

void setup_engine()
{
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
}

int check_vector(const vector<double>& a, float **b, const vector<int> &shuf, int batch_size, int batch_id, int current_batch_size, int col) {
    int i;

    for (i = 0; i < current_batch_size; ++i) {
        if (fabs(b[shuf[batch_id*batch_size+i]][col] - a[i]) > 0.001) {
            cout << "sample: " << shuf[batch_id*batch_size+i] << endl;
            cout << b[shuf[batch_id*batch_size+i]][col] << " v.s. " << a[i] << endl;
            return -1;
        }
    }
    
    return 0;
}

void test_read_and_write(){
    SEALCiphertext op1, op2;
    SEALCiphertext result;
    SEALPlaintext op1_p, op2_p, result_p;
    vector<double> op1_v{1,2,3,4,5};
    vector<double> op2_v{5,4,3,2,1};
    vector<double> result_v;
    engine->encode(op1_v, op1_p);
    engine->encode(op2_v, op2_p);
    engine->encrypt(op1_p, op1);
    engine->encrypt(op2_p, op2);
    // 1. save the engine 2. save the ciphertext
    SEALCiphertext array[] = {op1, op2};
    save_engine(engine, "engine.save");
    save_ciphertext(array, 2, "ciphertext.save");

    engine = make_shared<SEALEngine>();
    // 1. load the engine 2. load the ciphertext
    load_engine(engine, "engine.save");
    SEALCiphertext * array2 = new SEALCiphertext[2];
    load_ciphertext(array2, engine, 2, "ciphertext.save");
    seal_add(array2[0], array2[1], result);
    engine->decrypt(result, result_p);
    engine->decode(result_p, result_v);
    print_vector(result_v);    
}

int main(int argc, char *argv[])
{
    int i, j, ret_size;
    int seed = 11;
    int total_samples = 10, left_sample_num;
    int mini_size = 3;
    kann_data_t *data, *label;
    int batch_num, batch_size, current_batch_size;
    string base_dir, batch_dir;

    vector<int> shuf;
    vector<double> image_features;
    vector<double> image_labels;
    SEALPlaintext plain_tmp(engine);
    vector<SEALCiphertext> image_data_c;
    vector<SEALCiphertext> image_labels_c;


    if (argc != 3) {
        cout << "Usage: " << argv[0] << " data" << " label" << endl;
        return 0;
    }

    setup_engine();
    test_read_and_write();

/**
    kann_srand(seed);
    data = kann_data_read(argv[1]);
    label = kann_data_read(argv[2]);

    base_dir = "tmp";
    batch_num = shuffle_and_encrypt_dataset(total_samples, mini_size, data, label, base_dir, shuf);
    if (batch_num < 0)
        goto exit;
    
    cout << "save encrypted ciphertext in " << base_dir << endl;

    cout << "load ciphertext in batch" << endl;
    image_data_c.resize(data->n_col, SEALCiphertext(engine));
    image_labels_c.resize(label->n_col, SEALCiphertext(engine));
    left_sample_num = total_samples;
    batch_size = left_sample_num >= mini_size? mini_size : left_sample_num;
    for (i = 0; i < batch_num && left_sample_num > 0; ++i, left_sample_num -= batch_size) {
        batch_dir = base_dir + "/" + to_string(i);
        ret_size = load_batch_ciphertext(image_data_c, batch_dir, 0);
        if (ret_size < 0 || ret_size != data->n_col) {
            cout << "load data return " << ret_size << " at batch " << i << endl;
            goto exit;
        }
        ret_size = load_batch_ciphertext(image_labels_c, batch_dir, 1);
        if (ret_size < 0 || ret_size != label->n_col) {
            cout << "load label return " << ret_size << " at batch " << i << endl;
            goto exit;
        }
        // assert data integrity.
        current_batch_size = left_sample_num >= mini_size? mini_size : left_sample_num;
        image_features.clear();
        image_labels.clear();
        image_features.resize(current_batch_size);
        image_labels.resize(current_batch_size);
        for (j = 0; j < data->n_col; j++){
            // if (image_data_c[j].size() != current_batch_size) {
            //     cout << "[Error ] features vector in batch " << i << " col " << j
            //          << " has wrong size (" << image_data_c[j].size() << "), expect " 
            //          << current_batch_size << endl;
            //     goto exit;
            // }
            engine->decrypt(image_data_c[j], plain_tmp);
            engine->decode(plain_tmp,image_features);
            if(check_vector(image_features, data->x, shuf, batch_size, i, current_batch_size, j)) {
                cout << "[Error ] check features vector in batch " << i << " col " << j << endl;
                goto exit;
            }
        }
        for (j = 0; j < label->n_col; j++){
            // if (image_labels_c[j].size() != current_batch_size) {
            //     cout << "[Error ] labels vector in batch " << i << " col " << j
            //          << " has wrong size (" << image_labels_c[j].size() << "), expect " 
            //          << current_batch_size << endl;
            //     goto exit;
            // }
            engine->decrypt(image_labels_c[j], plain_tmp);
            engine->decode(plain_tmp,image_labels);
            if(check_vector(image_labels, label->x, shuf, batch_size, i, current_batch_size, j)) {
                cout << "[Error ] check labels vector in batch " << i << " col " << j << endl;
                goto exit;
            }
        }
    }
    cout << "load test pass" << endl;

exit:
    kann_data_free(data);
    kann_data_free(label);
**/
    return 0;
}