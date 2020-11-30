#ifndef UTIL_H
#define UTIL_H

#include "kann.h"
#include "kann_extra/kann_data.h"

using namespace std;

#define DATA_FILE_PREFIX	"data_"
#define LABEL_FILE_PREFIX	"label_"

template<typename T>
static inline void print_vector(std::vector<T> vec, size_t print_size = 4, int prec = 3);

void print_ciphertext(SEALCiphertext *cipher);

void print_model(kann_t * model, int from, bool grad);

// TODO: it should move to HEWrapper
void save_ciphertext(SEALCiphertext& ciphertext, string filename);

// TODO: it should move to HEWrapper
int load_ciphertext(SEALCiphertext& ciphertext, string filename);

// is_label: 0 load data, 1 load label
int load_batch_ciphertext(vector<SEALCiphertext>& ciphertext_vec, string dir, int is_label);

int shuffle_and_encrypt_dataset(int total_samples, int mini_size, kann_data_t *data, kann_data_t *label, string output_dir, vector<int> &shuf);

#endif