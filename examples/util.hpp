#include <cstdlib>
#include <unistd.h>
#include <cassert>
#include <cstdio>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream> // ifstream/ofstream
#include <sys/stat.h> // struct stat, stat, mkdir
#include <dirent.h> // struct dirent
#include <sys/types.h> // DIR
#include "kann.h"
#include "kann_extra/kann_data.h"

using namespace std;

#define DATA_FILE_PREFIX	"data_"
#define LABEL_FILE_PREFIX	"label_"

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
	int i,j;
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
				print_vector(t, 4, 10);
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
					print_vector(t, 4, 10);
				}
				if(grad){
					cout << "encrytped grad:" << endl;
					for (j = 0; j < kad_len(model->v[i]); j++){
						if(model->v[i]->g_c[j].clean()){
							cout << "clean grad" << endl;
						}else{
							engine->decrypt(model->v[i]->g_c[j], *plaintext);
							engine->decode(*plaintext, t);
							print_vector(t, 4, 10);
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
					for (j = 0; j < kad_len(model->v[i]); j++){
						cout << model->v[i]->g[j] << endl;
					}
				}
			}
		}
	}
}

// TODO: it should move to HEWrapper
void save_ciphertext(SEALCiphertext& ciphertext, string filename)
{
	ofstream ct;
	ct.open(filename, ios::binary);
	ciphertext.ciphertext().save(ct);
	ct.close();
};

// TODO: it should move to HEWrapper
int load_ciphertext(SEALCiphertext& ciphertext, string filename)
{
	ifstream ct;
  	ct.open(filename, ios::binary);
	if (!ct.is_open()) {
		cout << filename << " not exist" << endl;
		return -1;
	}
    // TODO: maybe some wrapper attributes should be set
  	ciphertext.ciphertext().load(*(engine->get_context()->get_sealcontext()), ct);
	ct.close();
	return 0;
}

// is_label: 0 load data, 1 load label
int load_batch_ciphertext(vector<SEALCiphertext>& ciphertext_vec, string dir, int is_label)
{
	int i;
	struct dirent *filename;
	string path;
	int data_num;
	struct stat s_buf;
	DIR *dp;

	stat(dir.c_str(),&s_buf);
	if (!S_ISDIR(s_buf.st_mode)) {
		cout << "dir " << dir << "not exist!" << endl;
		return -1;
	}

	ciphertext_vec.clear();
	data_num = 0;

	dp = opendir(dir.c_str());
	while((filename = readdir(dp))) {
		if ((is_label && strncasecmp(filename->d_name, LABEL_FILE_PREFIX, strlen(LABEL_FILE_PREFIX)) == 0)
			|| (!is_label && strncasecmp(filename->d_name, DATA_FILE_PREFIX, strlen(DATA_FILE_PREFIX)) == 0)) {
				++data_num;
		}
		else {
			continue;
		}
	}

	ciphertext_vec.resize(data_num);
	for (i = 0; i < data_num; ++i) {
		path = dir + "/" + (is_label ? LABEL_FILE_PREFIX : DATA_FILE_PREFIX) + to_string(i);
		if (load_ciphertext(ciphertext_vec[i], path)) {
			return -1;
		}
	}

	return data_num;
}

int shuffle_and_encrypt_dataset(int total_samples, int mini_size, kann_data_t *data, kann_data_t *label, string output_dir, vector<int> &shuf)
{
	int i, j, k, ret = -1;
	string batch_dir, filename;
	int left_sample_num, current_sample_id, current_cipher_id, cipher_num, batch_size;
	SEALPlaintext plain_tmp(engine);
	vector<vector<double>> image_features;
	vector<vector<double>> image_labels;

	struct stat s_buf;

	if (!data || !label) {
		cout << "data or label is null pointer" << endl;
		return -1;
	}

	if (total_samples == 0) {
		cout << "total samples is zero" << endl;
		return -1;
	}

	if (!stat(output_dir.c_str(),&s_buf)) {
		cout << output_dir << " already exists" << endl;
		return -1;
	}
    if (mkdir(output_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)) {
        cout << "fail to create dir " << output_dir << endl;
        return -1;
	}

	shuf.clear();
	shuf.reserve(total_samples);

	for (i = 0; i < total_samples; ++i) shuf[i] = i;
	kann_shuffle(total_samples, &shuf[0]);

	//batching: in secure training we can use different batch size for every batch. 
	left_sample_num = total_samples;
	current_sample_id = 0;
	current_cipher_id = 0;
	cipher_num = total_samples % mini_size == 0? total_samples / mini_size : total_samples / mini_size + 1;	
	SEALCiphertext cipher_tmp(engine);
	image_features.resize(data->n_col);
	image_labels.resize(label->n_col);

	while (left_sample_num > 0){
		batch_size = left_sample_num >= mini_size? mini_size : left_sample_num;
		for (j = 0; j < data->n_col; j++)
			image_features[j].resize(batch_size);
		for (j = 0; j < label->n_col; j++)
			image_labels[j].resize(batch_size);
		for (k = current_sample_id; k < current_sample_id + batch_size; k++){
			for (j = 0; j < data->n_col; j++){
				image_features[j][k%batch_size] = data->x[shuf[k]][j];			
			}
			for (j = 0; j < label->n_col; j++){
				image_labels[j][k] = label->x[shuf[k]][j];			
			}
		}

		batch_dir = output_dir + "/" + to_string(current_cipher_id);
		if (mkdir(batch_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)) {
			cout << "fail to create dir " << batch_dir << endl;
			goto exit;
		}
		for (j = 0; j < data->n_col; j++){
			engine->encode(image_features[j], plain_tmp);
			engine->encrypt(plain_tmp, cipher_tmp);
			filename = batch_dir + "/" + DATA_FILE_PREFIX + to_string(j);
			save_ciphertext(cipher_tmp, filename);
		}
		for (j = 0; j < label->n_col; j++){
			engine->encode(image_labels[j], plain_tmp);
			engine->encrypt(plain_tmp, cipher_tmp);
			filename = batch_dir + "/" + LABEL_FILE_PREFIX + to_string(j);
			save_ciphertext(cipher_tmp, filename);
		}
		current_cipher_id ++;
		current_sample_id += batch_size;
		left_sample_num -= batch_size;
	}
	ret = cipher_num;

exit:
	return ret;
}