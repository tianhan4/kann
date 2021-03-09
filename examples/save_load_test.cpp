#include <cstdlib>
#include <cmath>
#include <iostream>
#include <omp.h>
#include "NetIO.h"
#include "kann.h"
#include "kann_extra/kann_data.h"
#include "util.h"

static kann_t *model_gen(int n_in, int n_out, int loss_type, int n_h_layers, int n_h_neurons, float h_dropout, bool w_is_encrypted, bool b_is_encrypted)
{
	int i;
	kad_node_t *t;
	t = kann_layer_input(n_in);
	for (i = 0; i < n_h_layers; ++i)
		t = kad_relu(kann_layer_dense(t, n_h_neurons, w_is_encrypted));
		//t = kann_layer_dropout(kad_relu(kann_layer_dense(t, n_h_neurons, w_is_encrypted, b_is_encrypted)), h_dropout);
		 //better put the last layer all encrypted. For we can infer the i-1 activations from the gradients of w, due to the situation where only 1 node output.
	return kann_new(kann_layer_cost(t, n_out, loss_type, true, true), 0);
}

void setup_engine(int batch_size, const string& base_dir)
{
    int i;
    string engine_path;
    cout << "set up the encryption engine" << endl;
    size_t poly_modulus_degree = 8192;
	size_t standard_scale = 30;
    std::vector<int> coeff_modulus = {40, 30, 30, 30, 30, 40};
    SEALEncryptionParameters parms(poly_modulus_degree,
                    coeff_modulus,
                    seal_scheme::CKKS);
    engine = make_shared<SEALEngine>();

    engine_path = base_dir + "/" + ENGINE_FILE;
    if (load_engine(engine, engine_path) < 0) {
        cout << "laod engine fail in " << engine_path << endl;
        cout << "init engine from parms" << endl;
        engine->init(parms, standard_scale, false);
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
    engine->lazy_relinearization() = true;
    engine->lazy_mode() = 1;  
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


void prepareDataset(int total_samples, int mini_size, vector<vector<SEALCiphertext>> & features_c, vector<vector<SEALCiphertext>> & labels_c){
	// shuffling, batching and encrypting the data. (packaging into a cpp file later)
	SEALPlaintext tmp(engine);
    int i,j,k;
	int * shuf = (int*)malloc(total_samples * sizeof(int));
	for (i = 0; i < total_samples; ++i) shuf[i] = i;
    kann_shuffle(total_samples, shuf);	
	
	double train_cost;
	//create a pseudo dataset
	kann_data_t *pseudo_data = new kann_data_t();
	pseudo_data->n_row = total_samples;
	pseudo_data->n_col = 10;
	pseudo_data->x = new float* [pseudo_data->n_row];
	for (i = 0 ; i< pseudo_data->n_row; i++){
		pseudo_data->x[i] = new float [pseudo_data->n_col];
		for (j = 0; j < pseudo_data->n_col; j++){
			pseudo_data->x[i][j] = 1.0;
		}
	}

	kann_data_t *pseudo_label = new kann_data_t();
	pseudo_label->n_row = total_samples;
	pseudo_label->n_col = 1;
	pseudo_label->x = new float* [pseudo_label->n_row];
	for (i = 0 ; i< pseudo_label->n_row; i++){
		pseudo_label->x[i] = new float [pseudo_label->n_col];
		for (j = 0; j < pseudo_label->n_col; j++){
			pseudo_label->x[i][j] = 50.0;
		}
	}

	//kann_data_t *used_dataset = in;
	kann_data_t *used_dataset = pseudo_data;
	kann_data_t *used_label = pseudo_label;

	//batching: in secure training we can use different batch size for every batch. 
	int left_sample_num = total_samples;
	int current_sample_id = 0;
	int current_cipher_id = 0;
	int cipher_num = total_samples % mini_size == 0? total_samples / mini_size : total_samples / mini_size + 1;
	features_c.resize(cipher_num, vector<SEALCiphertext>(used_dataset->n_col, SEALCiphertext(engine)));
	labels_c.resize(cipher_num, vector<SEALCiphertext>(used_label->n_col, SEALCiphertext(engine)));
	vector<vector<double>> features;
	vector<vector<double>> labels;
	features.resize(used_dataset->n_col);
	labels.resize(used_label->n_col);
	while (left_sample_num > 0){
		int batch_size = left_sample_num >= mini_size? mini_size : left_sample_num;
		for (j = 0; j < used_dataset->n_col; j++)
			features[j].resize(batch_size);
		for (j = 0; j < used_label->n_col; j++)
			labels[j].resize(batch_size);
		for (k = current_sample_id; k < current_sample_id + batch_size; k++){
			for (j = 0; j < used_dataset->n_col; j++){
				features[j][k] = used_dataset->x[shuf[k]][j];			
			}
			for (j = 0; j < used_label->n_col; j++){
				labels[j][k] = used_label->x[shuf[k]][j];			
			}
		}
		for (j = 0; j < used_dataset->n_col; j++){
			engine->encode(features[j], tmp);
			engine->encrypt(tmp, features_c[current_cipher_id][j]);
		}
		for (j = 0; j < used_label->n_col; j++){
			engine->encode(labels[j], tmp);
			engine->encrypt(tmp, labels_c[current_cipher_id][j]);
		}
		current_cipher_id ++;
		current_sample_id += batch_size;
		left_sample_num -= batch_size;
	}
	assert(current_cipher_id == cipher_num);
	fprintf(stdout, "Finishing data encryption.\n");
	cout << "sample num:"<< current_sample_id << endl;
	cout << "cipher batch num:"<< current_cipher_id << endl;
}

void test_save_and_load(){
    // 1. Build the model
    kann_t * ann = model_gen(10, 1, KANN_C_MSE, 1, 32, 0, false, true);

    // 2. Prepare the dataset
    vector<vector<SEALCiphertext>> features_c, labels_c;
    prepareDataset(128, 8, features_c, labels_c);

    float train_cost, i_cost;
    // 3. Perform some training steps
    for (int i = 0; i < 3; i ++){
        kann_switch(ann, 1);
        SEALCiphertext * bind_data = features_c[i].data();
        SEALCiphertext * bind_label = labels_c[i].data();	
        kann_feed_bind(ann, KANN_F_IN, 0, &bind_data);
        kann_feed_bind(ann, KANN_F_TRUTH, 0, &bind_label);
        train_cost = kann_cost(ann, 0, 1);
        for (int k = 0; k < ann->n; k++){
            if (kad_is_var(ann->v[k])){
                if(seal_is_encrypted(ann->v[k]))
                    kann_SGD(kad_len(ann->v[k]), 0.01, 0, ann->v[k]->g_c, ann->v[k]->x_c);
                else
                    kann_SGD(kad_len(ann->v[k]), 0.01, 0, ann->v[k]->g, ann->v[k]->x);	
            }
        }
        cout << "step: " << i << ", " << "training cost: " << train_cost << endl;
    }

    //print_model(ann, i_cost, 1, 1);
    
    // 3.5 Do some inference, save the result
    kann_switch(ann, 0);
    SEALCiphertext * bind_data = features_c[5].data();	
	kann_feed_bind(ann, KANN_F_IN, 0, &bind_data);
    int i_out = kann_find(ann, KANN_F_OUT, 0);
	kad_eval_at(ann->n, ann->v, i_out);
    SEALCiphertext * inference_result = ann->v[i_out]->x_c;
    cout << "Print the inference result 1:" << endl;
    for(int i = 0; i < kad_len(ann->v[i_out]); i++)
        print_ciphertext(&inference_result[i]);

    // 4. Save the model
    kann_save("model.save", ann);

    // 5. Load the model and do the same inference
    kann_t * ann2 = kann_load("model.save");
    bind_data = features_c[5].data();	
	kann_feed_bind(ann2, KANN_F_IN, 0, &bind_data);
    i_out = kann_find(ann2, KANN_F_OUT, 0);
	kad_eval_at(ann2->n, ann2->v, i_out);
    SEALCiphertext * inference_result2 = ann2->v[i_out]->x_c;
    cout << "Print the inference result 2:" << endl;
    for(int i = 0; i < kad_len(ann2->v[i_out]); i++)
        print_ciphertext(&inference_result2[i]);

    // 6. compare the result, the test passes.
    vector<double> tmp_v1, tmp_v2;
    int is_same = true;
    for(int i = 0; i < kad_len(ann2->v[i_out]); i++){
        engine->decrypt(inference_result[i], *plaintext);
        engine->decode(*plaintext, tmp_v1);
        engine->decrypt(inference_result2[i], *plaintext);
        engine->decode(*plaintext, tmp_v2);
        for (int j = 0; j < inference_result[0].size(); j++){
            if(abs(tmp_v1[j] - tmp_v2[j]) < 1e-6)
                continue;
            else
            {
                is_same = false;
                goto result;
            }
            
        }
    }
result:
    cout << "Is same? Answer: " <<  bool(is_same) << endl;  
}

// Test model save/load.
// Firstly train a model for one step, then record the cost, then save it, then load it, then recode and compare the cost. If the two cost values are the same. The test passes.
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

    setup_engine(8, "");
    test_save_and_load();
    return 0;
}