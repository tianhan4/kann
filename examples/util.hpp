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
