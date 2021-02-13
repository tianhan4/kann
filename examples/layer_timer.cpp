#include <cstdlib>
#include <cmath>
#include <iostream>
#include <omp.h>
#include "kann.h"
#include "kann_extra/kann_data.h"
#include "util.h"

using namespace std;

int test_bandwidth_download = 12; //MBit/s
int test_bandwidth_uploadc= 6; //MBit/s


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

	//engine->zero = new SEALCiphertext(engine);
	//engine->encode(0, *plaintext);
	//engine->encrypt(*plaintext, *(engine->zero));
    engine->zero = nullptr;
    engine->lazy_relinearization() = false;
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
    lenet = lenet_gen(label_size);
    int repeat = 1;
    vector<pair<int, string>> recorded_layers;
    vector<int> forward_times, backward_times, other_times;
    int fp_time, bp_time, step_time = 0;
    time_layer(lenet, repeat, recorded_layers, forward_times, backward_times, other_times, fp_time, bp_time, step_time, base_dir, data_size, label_size);
    // print the results
    cout << "Per-component running results:" << endl;
    cout << "Repeat time: " << repeat << endl;
    cout << "Forwarding time:" << endl;
    for (i = 0; i < forward_times.size(); i++)
        cout << "   " << recorded_layers[i].second << ":" << forward_times[i] << "ms" << endl;
    cout << "Backwarding time:" << endl;
    for (i = 0; i < backward_times.size(); i++)
        cout << "   " << recorded_layers[i].second << ":" << backward_times[i] <<  "ms" << endl;
    cout << "Aggregation time:" << other_times[0] <<  "ms" << endl;
    cout << "Update time:" << other_times[1] <<  "ms" << endl;
    cout << "Forward time:" << fp_time << "ms" << endl;
    cout << "Backward time:" << bp_time << "ms" << endl;
    cout << "Step time:" << step_time << "ms" << endl;

    int i_cost = kann_find(lenet, KANN_F_COST, 0);
    print_model(lenet, i_cost, false, true);
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
