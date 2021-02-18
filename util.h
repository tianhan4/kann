#ifndef UTIL_H
#define UTIL_H

#include "kann.h"
#include "kann_extra/kann_data.h"

using namespace std;

#define ENGINE_FILE "engine.save"
#define DATA_FILE	"data"
#define LABEL_FILE	"label"

/**
void load_from_pb_tensor(float *target, pb::HETensor * pb_tensor, size_t size, std::shared_ptr<SEALEngine> engine);

pb::HETensor write_to_pb_tensors(float *source, size_t size);

void load_from_pb_tensor(std::vector<SEALCiphertext> &target, size_t offset, pb::HETensor * pb_tensor, std::shared_ptr<SEALEngine> engine);

pb::HETensor write_to_pb_tensors(std::vector<SEALCiphertext> &source, size_t offset, size_t size, std::shared_ptr<SEALEngine> engine);

void load_from_pb_tensor(SEALCiphertext * target, size_t offset, pb::HETensor * pb_tensor, std::shared_ptr<SEALEngine> engine);

pb::HETensor write_to_pb_tensors(SEALCiphertext* source, size_t offset, size_t size, std::shared_ptr<SEALEngine> engine);
**/
template <typename T>
inline std::unordered_map<std::string, std::pair<std::string, std::vector<double>>>
map_to_double_map(
    const std::unordered_map<std::string, std::pair<std::string, std::vector<T>>>& inputs);

template<typename T>
inline void print_vector(std::vector<T> vec, size_t print_size = 4, int prec = 3);

void print_ciphertext(SEALCiphertext *cipher);

void print_model(kann_t * model, int from, bool data, bool grad);

// is_label: 0 load data, 1 load label
int load_batch_ciphertext(vector<SEALCiphertext>& ciphertext_vec, int size, string dir, int is_label);

int shuffle_and_encrypt_dataset(int total_samples, int mini_size, kann_data_t *data, kann_data_t *label, string output_dir, vector<int> &shuf);

void save_engine(std::shared_ptr<hewrapper::SEALEngine> engine, string filename, bool is_rotate = true, bool is_decrypt = true);

int load_engine(std::shared_ptr<hewrapper::SEALEngine> engine, string filename);

void save_ciphertext(SEALCiphertext* ciphertext, size_t cipher_num, string filename);

int load_ciphertext(SEALCiphertext* ciphertext, std::shared_ptr<hewrapper::SEALEngine> engine, size_t cipher_num, string filename);


#endif