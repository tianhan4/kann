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
#include "util.h"

using namespace std;


/**
// load ciphertext/raw values from pb tensors
void load_from_pb_tensor(float *target, pb::HETensor * pb_tensor, size_t size, std::shared_ptr<SEALEngine> engine){
	assert(target != nullptr && "HETensor is empty");
	const auto* raw_data = pb_tensor->mutable_data()->Mutable(0)->mutable_plain()->mutable_data();
	std::memcpy(target, raw_data, sizeof(float) * size);
}

// write to pb tensors from ciphertext/raw values
pb::HETensor write_to_pb_tensors(float *source, size_t size){
	pb::HETensor pb_tensor;
	// Populate attributes of tensor to estimate byte size
	if (size>0) {
		auto* mutable_data = pb_tensor.mutable_data();
		mutable_data->Add();
		mutable_data->Mutable(0)->mutable_plain()->Reserve(size);
		auto* mutable_plain= mutable_data->Mutable(0)->mutable_plain()->mutable_data();
		std::memcpy(mutable_plain, source, sizeof(float) * size);
	}
	return pb_tensor;
}

// load ciphertext/raw values from pb tensors
void load_from_pb_tensor(std::vector<SEALCiphertext> &target, size_t offset, pb::HETensor * pb_tensor, std::shared_ptr<SEALEngine> engine){
	size_t result_count = pb_tensor->data_size();
	target.reserve(offset + result_count);
	#pragma omp parallel for
	// NOLINTNEXTLINE
	for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
		istringstream str(pb_tensor->data(result_idx).ciphertext());
		target[result_idx + offset].load(str, engine);
		target[result_idx + offset].init(engine);
	}
}

// write to pb tensors from ciphertext/raw values
pb::HETensor write_to_pb_tensors(std::vector<SEALCiphertext> &source, size_t offset, size_t size, std::shared_ptr<SEALEngine> engine){
	pb::HETensor pb_tensors;
	// Populate attributes of tensor to estimate byte size
	if (size>0) {
		auto* mutable_data = pb_tensors.mutable_data();
		for (size_t data_idx = 0; data_idx < size; ++data_idx) {
			mutable_data->Add();
		}
		#pragma omp parallel for
		// NOLINTNEXTLINE
		for (size_t data_idx = 0; data_idx < size; ++data_idx) {
			ostringstream str(*mutable_data->Mutable(data_idx)->mutable_ciphertext());
			source[data_idx + offset].save(str);
		}
	}
	return pb_tensors;
}

// load ciphertext/raw values from pb tensors
void load_from_pb_tensor(SEALCiphertext *target, size_t offset, pb::HETensor * pb_tensor, std::shared_ptr<SEALEngine> engine){
	size_t result_count = pb_tensor->data_size();
	#pragma omp parallel for
	// NOLINTNEXTLINE
	for (size_t result_idx = 0; result_idx < result_count; ++result_idx) {
		istringstream str(pb_tensor->data(result_idx).ciphertext());
		target[result_idx + offset].load(str, engine);
		target[result_idx + offset].init(engine);
	}
}

// write to pb tensors from ciphertext/raw values
pb::HETensor write_to_pb_tensors(SEALCiphertext *source, size_t offset, size_t size, std::shared_ptr<SEALEngine> engine){
	pb::HETensor pb_tensors;
	// Populate attributes of tensor to estimate byte size
	if (size>0) {
		auto* mutable_data = pb_tensors.mutable_data();
		for (size_t data_idx = 0; data_idx < size; ++data_idx) {
			mutable_data->Add();
		}
		#pragma omp parallel for
		// NOLINTNEXTLINE
		for (size_t data_idx = 0; data_idx < size; ++data_idx) {
			ostringstream str(*mutable_data->Mutable(data_idx)->mutable_ciphertext());
			source[data_idx + offset].save(str);
		}
	}
	return pb_tensors;
}
**/
template <typename T>
inline std::unordered_map<std::string,
                          std::pair<std::string, std::vector<double>>>
map_to_double_map(
    const std::unordered_map<std::string,
                             std::pair<std::string, std::vector<T>>>& inputs) {
  std::unordered_map<std::string, std::pair<std::string, std::vector<double>>>
      outputs;

  for (const auto& elem : inputs) {
    std::vector<double> double_inputs{elem.second.second.begin(),
                                      elem.second.second.end()};
    outputs.insert(
        {elem.first, std::make_pair(elem.second.first, double_inputs)});
  }
  return outputs;
}

void save_engine(std::shared_ptr<hewrapper::SEALEngine> engine, string filename, bool is_rotate, bool is_decrypt){
	
	ofstream en;
	en.open(filename, ios::binary);
	engine->save(en, is_rotate, is_decrypt);
	en.close();
}

int load_engine(std::shared_ptr<hewrapper::SEALEngine> engine, string filename){

	ifstream en;
	en.open(filename, ios::binary);
	if (!en.is_open()) {
		cout << filename << " not exist" << endl;
		return -1;
	}
	engine->load(en);
	en.close();
	return 1;
}

// Save all things in the ciphertext
void save_ciphertext(SEALCiphertext* ciphertext, size_t cipher_num, string filename)
{
	ofstream ct;
	int i;
	if(!ciphertext)
		return;
	ct.open(filename, ios::binary);
	for(i = 0; i < cipher_num; i++){
		ciphertext[i].save(ct);
	}
	ct.close();
};

//Load all things
int load_ciphertext(SEALCiphertext* ciphertext, std::shared_ptr<hewrapper::SEALEngine> engine, size_t cipher_num, string filename)
{
	ifstream ct;
	int i;
  	ct.open(filename, ios::binary);
	if (!ct.is_open()) {
		cout << filename << " not exist" << endl;
		return -1;
	}
	for(i = 0; i < cipher_num; i++){
		ciphertext[i].load(ct, engine);
		ciphertext[i].init(engine);
	}
	ct.close();
	
	return 0;
}

template<typename T>
inline void print_vector(std::vector<T> vec, size_t print_size, int prec)
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
	vector<double> output;
	engine->decrypt(*cipher, *plaintext);
	engine->decode(*plaintext, output);
	cout << output.size() << endl;
	print_vector(output);
}

/**
 * Print model information.
 * int from: print nodes from 0 to the node "from";
 * bool data: weather print the data values;
 * bool grad: weather print the gradient information;
 * ps: the values of gradients will be printed when and only when the data and grad are both true.
 * */
void print_model(kann_t * model, int from, bool data, bool grad){
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
			if (data)
				for (j = 0; j < kad_len(model->v[i]); j++){
					engine->decrypt(model->v[i]->x_c[j], *plaintext);
					engine->decode(*plaintext, t[0]);
					print_vector(t[0], 4, 10);
				}
		}
		else if (kad_is_back(model->v[i])){
			if(seal_is_encrypted(model->v[i])){
				cout << "encrypted:" << endl;
				cout << " level: " << engine->get_context()->get_sealcontext()->get_context_data(model->v[i]->x_c[0].ciphertext().parms_id())->chain_index() << endl;
				cout << "ciphertext size:" << model->v[i]->x_c[0].size() << endl;
				if (data)
					for (j = 0; j < kad_len(model->v[i]); j++){
						engine->decrypt(model->v[i]->x_c[j], *plaintext);
						engine->decode(*plaintext, t[0]);
						print_vector(t[0], 4, 10);
					}
				if(grad  && !(model->v[i]->ext_flag & KANN_F_COST)){
					cout << "encrytped grad:" << endl;
					int first_non_clean = -1;
					for (j = 0; j < kad_len(model->v[i]); j++){
						if(model->v[i]->g_c[j].clean()){
							if(data)
								cout << "clean grad" << endl;
						}else{
							first_non_clean = j;
							if(data){
								engine->decrypt(model->v[i]->g_c[j], *plaintext);
								engine->decode(*plaintext, t[0]);
								print_vector(t[0], 4, 10);
							}
						}
					}
					if(first_non_clean >=0 )
						cout << " level: " << engine->get_context()->get_sealcontext()->get_context_data(model->v[i]->g_c[first_non_clean].ciphertext().parms_id())->chain_index() << endl;
		
				}
			}else{
				cout << "plain var:" << endl;
				if(data)
					for (j = 0; j < kad_len(model->v[i]); j++){
						cout << model->v[i]->x[j] << endl;
					}
				if (grad){
					cout << "plain grad:" << endl;
					if(data)
						for (j = 0; j < kad_len(model->v[i]); j++){
							cout << model->v[i]->g[j] << endl;
						}
				}
			}
		}
	}
}

// is_label: 0 load data, 1 load label
int load_batch_ciphertext(vector<SEALCiphertext>& ciphertext_vec, int size, string dir, int is_label)
{
	string path;
	struct stat s_buf;

	stat(dir.c_str(),&s_buf);
	if (!S_ISDIR(s_buf.st_mode)) {
		cout << "dir " << dir << " not exist!" << endl;
		return -1;
	}

	ciphertext_vec.clear();
	ciphertext_vec.resize(size);

	if (is_label) {
		path = dir + '/' + LABEL_FILE;
	}
	else {
		path = dir + '/' + DATA_FILE;
	}

	load_ciphertext(ciphertext_vec.data(), engine, size, path);
	
	return size;
}

int batch_save(int left_sample_num, int mini_size, int& current_cipher_id, int& current_sample_id,
				const vector<int> &shuf, const string& output_dir, kann_data_t *data, kann_data_t *label)
{
	int j, k;
	int batch_size;
	vector<SEALCiphertext> cipher_data, cipher_label;
	SEALPlaintext plain_data(engine), plain_label(engine);
	string batch_dir, filename;
	vector<vector<double>> image_features(data->n_col);
	vector<vector<double>> image_labels(label->n_col);
	cout << "data row: " << data->n_row << " col: " << data->n_col << endl;
	cout << "label row: " << label->n_row << " col: " << label->n_col << endl;

	while (left_sample_num > 0){
		batch_size = left_sample_num >= mini_size? mini_size : left_sample_num;
		for (j = 0; j < data->n_col; j++)
			image_features[j].resize(batch_size);
		for (j = 0; j < label->n_col; j++)
			image_labels[j].resize(batch_size);
		for (k = current_sample_id; k < current_sample_id + batch_size; k++){
			for (j = 0; j < data->n_col; j++){
				image_features[j][k%mini_size] = data->x[shuf[k]][j];			
			}
			for (j = 0; j < label->n_col; j++){
				image_labels[j][k%mini_size] = label->x[shuf[k]][j];			
			}
		}

		batch_dir = output_dir + "/" + to_string(current_cipher_id);
		if (mkdir(batch_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)) {
			cout << "fail to create dir " << batch_dir << endl;
			return -1;
		}

		cipher_data.resize(data->n_col, SEALCiphertext(engine));
		for (j = 0; j < data->n_col; j++){
			engine->encode(image_features[j], plain_data);
			engine->encrypt(plain_data, cipher_data[j]);
			// filename = batch_dir + "/" + DATA_FILE_PREFIX + to_string(j);
			// save_ciphertext(&cipher_tmp, 1, filename);
		}
		filename = batch_dir + "/" + DATA_FILE;
		save_ciphertext(cipher_data.data(), data->n_col, filename);

		cipher_label.resize(label->n_col, SEALCiphertext(engine));
		for (j = 0; j < label->n_col; j++){
			engine->encode(image_labels[j], plain_label);
			engine->encrypt(plain_label, cipher_label[j]);
			// filename = batch_dir + "/" + LABEL_FILE_PREFIX + to_string(j);
			// save_ciphertext(&cipher_tmp, 1, filename);
		}
		filename = batch_dir + "/" + LABEL_FILE;
		save_ciphertext(cipher_label.data(), label->n_col, filename);
		current_cipher_id ++;
		current_sample_id += batch_size;
		left_sample_num -= batch_size;
	}
	return 0;
}

int shuffle_and_encrypt_dataset(int total_samples, int mini_size, kann_data_t *data, kann_data_t *label,
								string output_dir, vector<int> &shuf)
{
	int i, ret = -1;
	int left_sample_num, current_sample_id, current_cipher_id, cipher_num, sample_size;
	string engine_file;

	struct stat s_buf;

	if (!data || !label) {
		cout << "data or label is null pointer" << endl;
		return -1;
	}

	if (total_samples == 0) {
		cout << "total samples is zero" << endl;
		return -1;
	}

	cipher_num = total_samples % mini_size == 0? total_samples / mini_size : total_samples / mini_size + 1;
	if (!stat(output_dir.c_str(),&s_buf)) {
		cout << output_dir << " already exists" << endl;
		return cipher_num;
	}
    if (mkdir(output_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)) {
        cout << "fail to create dir " << output_dir << endl;
        return -1;
	}

	shuf.clear();
	shuf.reserve(total_samples);

	for (i = 0; i < total_samples; ++i) shuf[i] = i;
	// kann_shuffle(total_samples, &shuf[0]);

	engine_file = output_dir + "/" + ENGINE_FILE;
	save_engine(engine, engine_file);

	//batching: in secure training we can use different batch size for every batch. 
	left_sample_num = total_samples;
	current_sample_id = 0;
	current_cipher_id = 0;	

	while (left_sample_num > 0) {
		sample_size = left_sample_num > 10 * mini_size ? 10 * mini_size : left_sample_num;
		if (batch_save(sample_size, mini_size, current_cipher_id, current_sample_id,
					shuf, output_dir, data, label)) {
			goto exit;
		}
		left_sample_num -= sample_size;
	}
	
	ret = cipher_num;

exit:
	return ret;
}