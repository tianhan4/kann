#include <cmath>
#include <cfloat>
#include <cstring>
#include <cstdlib>	
#include <cassert>
#include <cstdarg>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include "kann.h"
#include "util.h"

int kann_verbose = 3;
extern bool remote;
extern std::shared_ptr<SEALEngine> engine;
extern SEALPlaintext *plaintext;
extern SEALCiphertext *ciphertext;
extern vector<double> *t;
extern vector<double> *test_t;   
extern vector<double> *truth_t;

/******************************************
 *** @@BASIC: fundamental KANN routines ***
 ******************************************/


//providing uniform memory for a model's leaves.
static void kad_ext_sync(int n, kad_node_t **a, float *c, float *x, SEALCiphertext *x_c, float *g, SEALCiphertext * g_c)
{
	int i, j, k, j_c, k_c, j_g, l, m;
	for (i = j = k = j_c = k_c = j_g = 0; i < n; ++i) {
		kad_node_t *v = a[i];
		if (kad_is_var(v)) {
			l = kad_len(v);
			cout << "allocate var node:" << i << "len" << l << "encrypted" << seal_is_encrypted(v) <<endl;
			if(seal_is_encrypted(v)){
				if(v->x_c){
					for (m = 0; m < l; m ++){
						x_c[j_c+m] = v->x_c[m];
					}
					delete[] v->x_c;
				}
				v->x_c = &x_c[j_c];
				j_c += l;
			}else{
				if(v->x){
					std::memcpy(&x[j], v->x, l * sizeof(float));
					delete[] v->x;
				}
				v->x = &x[j];
				j += l;
			}
			v->g = &g[j_g];
			v->g_c = &g_c[j_g];
			j_g += l;
		} else if (kad_is_const(v)) {
			l = kad_len(v);
			if(v->x){
				std::memcpy(&c[k], v->x, l * sizeof(float));
				delete[] v->x;
			}
			v->x = &c[k];
			k += l;
		}
	}
}


//rerange the leaves' arrays into three arrays: const, gradient, x, g_c, x_c
//work together with kann_new_leaf_array
static void kad_ext_collate(int n, kad_node_t **a, float **_x, float **_g, float **_c, SEALCiphertext ** _x_c, 
SEALCiphertext ** _g_c)
{
	int i, j, k, j_c, k_c, j_g, l, m;
	float *x, *g, *c;
	SEALCiphertext *x_c, *g_c, *c_c;
	int n_var = kad_size_var(n, a);
	int n_encrypted_var = kad_size_encrypted_var(n, a);
	int n_unencrypted_var = n_var - n_encrypted_var;
	int n_const = kad_size_const(n, a);
	cout << "all variables:" << n_var << endl;
	cout << "encrypted variables:" << n_encrypted_var << endl;
	cout << "const: " << n_const << endl;
	if(*_x) delete[] *_x;
	x = *_x = new float[n_unencrypted_var];
	//we assume all const values are plain.
	if(*_c) delete[] *_c;
	c = *_c = new float[n_const];
	if(*_x_c) delete[] *_x_c;
	x_c = *_x_c = new SEALCiphertext[n_encrypted_var];
	
	if(*_g) delete[] *_g;
	g = *_g = new float[n_var];
	if(*_g_c) delete[] *_g_c;
	g_c = *_g_c = new SEALCiphertext[n_var];

	memset(g, 0, n_var * sizeof(float));
	for(i = 0; i < n_var; i++) {
		g_c[i].init(engine);
		g_c[i].clean() = true;
	}

	kad_ext_sync(n, a, c, x, x_c, g, g_c);
}



// From the cost node, including all roots altogether, build the model.
kann_t *kann_new(kad_node_t *cost, int n_rest, ...)
{
	kann_t *a;
	int i, n_roots = 1 + n_rest, has_pivot = 0, has_recur = 0;
	kad_node_t **roots;
	va_list ap;

	if (cost->n_d != 0) return 0;

	va_start(ap, n_rest);
	roots = (kad_node_t**)malloc((n_roots + 1) * sizeof(kad_node_t*));
	for (i = 0; i < n_rest; ++i)
		roots[i] = va_arg(ap, kad_node_t*);
	roots[i++] = cost;
	va_end(ap);

	cost->ext_flag |= KANN_F_COST;
	a = (kann_t*)calloc(1, sizeof(kann_t));
	a->v = kad_compile_array(&a->n, n_roots, roots);

	for (i = 0; i < a->n; ++i) {
		if (a->v[i]->pre) has_recur = 1;
		if (kad_is_pivot(a->v[i])) has_pivot = 1;
	}
	if (has_recur && !has_pivot) { /* an RNN that doesn't have a pivot; then add a pivot on top of cost and recompile */
		cost->ext_flag &= ~KANN_F_COST;
		roots[n_roots-1] = cost = kad_avg(1, &cost), cost->ext_flag |= KANN_F_COST;
		std::free(a->v);
		a->v = kad_compile_array(&a->n, n_roots, roots);
	}
	kad_ext_collate(a->n, a->v, &a->x, &a->g, &a->c, &a->x_c, &a->g_c);
	std::free(roots);
	return a;
}



void kann_delete_unrolled(kann_t *a)
{
	if (a && a->mt) kann_mt(a, 0, 0);
	if (a && a->v) kad_delete(a->n, a->v);
	free(a);
}

void kann_delete(kann_t *a)
{
	if (a == 0) return;
	delete[] a->x;
	delete[] a->g;
	delete[] a->c;
	delete[] a->x_c;
	delete[] a->g_c;
	kann_delete_unrolled(a);
}


static void kann_switch_core(kann_t *a, int is_train)
{
	int i;
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->op == 12 && a->v[i]->n_child == 2)
			*(int32_t*)a->v[i]->ptr = !!is_train;
}

//check flag 
#define chk_flg(flag, mask) ((mask) == 0 || ((flag) & (mask)))
//check label
#define chk_lbl(label, query) ((query) == 0 || (label) == (query))

// -1 return: not found. -2 return: multiple found.
int kann_find(const kann_t *a, uint32_t ext_flag, int32_t ext_label)
{
	int i, k, r = -1;
	for (i = k = 0; i < a->n; ++i)
		if (chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
			++k, r = i;
	return k == 1? r : k == 0? -1 : -2;
}

int kann_feed_bind(kann_t *a, uint32_t ext_flag, int32_t ext_label, SEALCiphertext **x_c)
{
	int i, k;
	if (x_c == 0) return 0;
	for (i = k = 0; i < a->n; ++i)
		if (kad_is_feed(a->v[i]) && chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label)){
			// cout << "binding input into node:" << i << endl;
			a->v[i]->x_c = x_c[k++];
		}
	return k;
}

int kann_feed_dim(const kann_t *a, uint32_t ext_flag, int32_t ext_label)
{  
	int i, k, n = 0;
	for (i = k = 0; i < a->n; ++i)
		if (kad_is_feed(a->v[i]) && chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
			++k, n = a->v[i]->n_d > 1? kad_len(a->v[i]) : a->v[i]->n_d == 1? a->v[i]->d[0] : 1;
	return k == 1? n : k == 0? -1 : -2;
}

static float kann_cost_core(kann_t *a, int cost_label, int cal_grad, double lr)
{
	int i_cost;
	SEALCiphertext cost_c(engine);
	float cost;
	i_cost = kann_find(a, KANN_F_COST, cost_label);
	assert(i_cost >= 0);
	cost_c = *kad_eval_at(a->n, a->v, i_cost);
	if (cal_grad) kad_grad(a->n, a->v, i_cost, engine->noise_mode(), lr);
	hewrapper::sum_vector(cost_c);
	if (remote){
		assert(false);
	}else{
		engine->decrypt(cost_c, *plaintext);
		engine->decode(*plaintext, t[0]);
		cost = t[0][0];
	}
	return cost;
}

int kann_eval(kann_t *a, uint32_t ext_flag, int ext_label)
{
	int i, k;
	for (i = k = 0; i < a->n; ++i)
		if (chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
			++k, a->v[i]->tmp = 1;
	kad_eval_marked(a->n, a->v);
	return k;
}


// used for debug. return the number errors. base means the total checked number.
static int kann_class_error_core(const kann_t *ann, float *truth, int *base)
{
	int i, j, k, m, n, off, n_err = 0;
	vector<vector<double>> prob_vectors;
	for (i = 0, *base = 0; i < ann->n; ++i) {
		kad_node_t *p = ann->v[i];
		if (((p->op == 13 && (p->n_child == 2 || p->n_child == 3)) || (p->op == 22 && p->n_child == 2)) && p->n_d == 0) { /* ce_bin or ce_multi */
			kad_node_t *x = p->child[0];
			n = x->d[x->n_d - 1];
			
			for (k = 0; k < n; ++k) {
				vector<double> tmp;
				engine->decrypt(x->x_c[k], *plaintext);
				engine->decode(*plaintext, tmp); 
				prob_vectors.push_back(tmp);
			}
			for(j = 0; j < x->x_c[0].size(); j++){
				float t_sum = 0.0f, t_min = 1.0f, t_max = 0.0f, x_max = 0.0f, x_min = 1.0f;
				int x_max_k = -1, t_max_k = -1;
				for (k = 0; k < n; ++k) {
					float xk = prob_vectors[k][j], tk = truth[j*n + k];
					t_sum += tk;
					t_min = t_min < tk? t_min : tk;
					x_min = x_min < xk? x_min : xk;
					if (t_max < tk) t_max = tk, t_max_k = k;
					if (x_max < xk) x_max = xk, x_max_k = k;
				}
				if (abs(t_sum - 1.0f) <= 0.0001f && t_min >= 0.0f && x_min >= 0.0f && x_max <= 1.0f) {
					++(*base);
					n_err += (x_max_k != t_max_k);
					// cout << "predited class:" << x_max_k << "truth class:" << t_max_k << "error:" << (x_max_k != t_max_k);
				}else{
					throw invalid_argument("Invalid sample.");
				}
			}
		}
	}
	return n_err;
}


void kann_mt(kann_t *ann, int n_threads, int max_batch_size) {}
float kann_cost(kann_t *a, int cost_label, int cal_grad, double lr) { return kann_cost_core(a, cost_label, cal_grad, lr); }
int kann_eval_out(kann_t *a) { return kann_eval(a, KANN_F_OUT, 0); }
int kann_class_error(const kann_t *a, float *truth, int *base) { return kann_class_error_core(a, truth, base); }
void kann_switch(kann_t *ann, int is_train) { return kann_switch_core(ann, is_train); }

/***********************
 *** @@IO: model I/O ***
 ***********************/

#define KANN_MAGIC "KAN\1"

void kann_save_fp(ostream & fs, kann_t *ann)
{
	fs.write(KANN_MAGIC, 4);
	kad_save(fs, ann->n, ann->v);
	engine->save(fs, true, true);

	int n_var = kad_size_var(ann->n, ann->v);
	int n_encrypted_var = kad_size_encrypted_var(ann->n, ann->v);
	int n_unencrypted_var = n_var - n_encrypted_var;
	int n_const = kad_size_const(ann->n, ann->v);
	fs.write((char *) ann->x, sizeof(float) * n_unencrypted_var);
	fs.write((char *) ann->c, sizeof(float) * n_const);
	for(int i = 0; i < n_encrypted_var; i++){
		ann->x_c[i].save(fs);
	}
}

void kann_save(const char *fn, kann_t *ann){
	fstream fs(fn, std::ios::binary | std::ios::out);
	kann_save_fp(fs, ann);
}

kann_t *kann_load_fp(istream & fs)
{
	char magic[4];
	kann_t *ann;
	int i;

	fs.read(magic, 4);
	if (strncmp(magic, KANN_MAGIC, 4) != 0) {
		return 0;
	}
	ann = (kann_t*)calloc(1, sizeof(kann_t));
	ann->v = kad_load(fs, &ann->n);
	engine->load(fs);
	
	int n_var = kad_size_var(ann->n, ann->v);
	int n_encrypted_var = kad_size_encrypted_var(ann->n, ann->v);
	int n_unencrypted_var = n_var - n_encrypted_var;
	int n_const = kad_size_const(ann->n, ann->v);
	if(ann->x) delete[] ann->x;
	ann->x = new float[n_unencrypted_var];
	//we assume all const values are plain.
	if(ann->c) delete[] ann->c;
	ann->c = new float[n_const];
	if(ann->x_c) delete[] ann->x_c;
	ann->x_c = new SEALCiphertext[n_encrypted_var];
	if(ann->g) delete[] ann->g;
	ann->g = new float[n_var];
	if(ann->g_c) delete[] ann->g_c;
	ann->g_c = new SEALCiphertext[n_var];

	memset(ann->g, 0, n_var * sizeof(float));
	for(i = 0; i < n_var; i++) {
		ann->g_c[i].init(engine);
		ann->g_c[i].clean() = true;
	}

	fs.read((char *)ann->x, sizeof(float) * n_unencrypted_var);
	fs.read((char *)ann->c, sizeof(float) * n_const);
	for(int i = 0; i < n_encrypted_var; i++){
		ann->x_c[i].load(fs, engine);
		ann->x_c[i].init(engine);
	}

	cout << "Allocate extenal memory for model." << endl;
	kad_ext_sync(ann->n, ann->v, ann->c, ann->x, ann->x_c, ann->g, ann->g_c);

	int req_alloc = 0;
	for (i = 0; i < ann->n; ++i)
		if (ann->v[i]->n_child > 0 && ann->v[i]->x == 0 && ann->v[i]->x_c == 0) req_alloc = 1;
	if (req_alloc) {	
		cout << "Allocate internal memory for model." << endl;
		kad_allocate_internal(ann->n, ann->v);
	}

	return ann;
}

kann_t *kann_load(const char *fn)
{
	std::fstream fs(fn, std::ios::binary | std::ios::in);
	kann_t *ann;
	ann = kann_load_fp(fs);
	return ann;
}

/**********************************************
 *** @@LAYER: layers and model generation ***
 **********************************************/

/********** General but more complex APIs **********/

//create a new leaf, allocate its memory, and push it back into the leaf list.
//offset and par used when we also want to maintain the leaf nodes in a seperate list.
//Don't need to pre-define the batchsize, we can fill the value into all slots in the ciphertext.
//Only temporarily allocate x/x_c and initialize them, work together with kad_ext_collate to reallocate x/g later.
kad_node_t *kann_new_leaf_array(int *offset, kad_node_p *par, uint8_t flag, float x0_01, int n_d, int32_t d[KAD_MAX_DIM], bool is_encrypted)
{
	int i, j, len, off = offset && par? *offset : -1;
	kad_node_t *p;
	if (off >= 0 && par[off]) return par[(*offset)++];
	p = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	p->n_d = n_d, p->flag = flag;
	std::memcpy(p->d, d, n_d * sizeof(int32_t));
	len = kad_len(p);
	cout << "add a new leaf (n_d, encrypted)" << n_d << "," << is_encrypted << ", len:" << len <<endl;
	//allocate x/x_c
	if (is_encrypted){
		p->flag |= SEAL_CIPHER;
		p->x_c = new SEALCiphertext[len];
		if (p->n_d <= 1) {
			double sdev_inv;
			sdev_inv = 1.0 / sqrt((double)len);
			for (i = 0; i < len; ++i){
				p->x_c[i].init(engine);
				engine->encode((float)(kad_drand_normal(0) * sdev_inv), *plaintext);
				engine->encrypt(*plaintext, p->x_c[i]);
				// change to mod 1.
				std::shared_ptr<seal::SEALContext> context = engine->get_context()->get_sealcontext();
				auto data_level_1 = context->last_context_data()->prev_context_data();
				//cout << "data level:" << data_level_1->chain_index() << endl;
				engine->get_evaluator()->mod_switch_to_inplace(p->x_c[i].ciphertext(), data_level_1->parms_id());
				
			}
			/**
			for (i = 0; i < len; ++i){
				p->x_c[i].init(engine); 
				engine->encode(x0_01, *plaintext);
				engine->encrypt(*plaintext, p->x_c[i]);
			}
			**/
		} else {
			double sdev_inv;
			sdev_inv = 1.0 / sqrt((double)len) / p->d[0];
			for (i = 0; i < len; ++i){
				p->x_c[i].init(engine);
				engine->encode((float)(kad_drand_normal(0) * sdev_inv), *plaintext);
				engine->encrypt(*plaintext, p->x_c[i]);
				std::shared_ptr<seal::SEALContext> context = engine->get_context()->get_sealcontext();
				auto data_level_1 = context->last_context_data()->prev_context_data();
				//cout << "data level:" << data_level_1->chain_index() << endl;
				engine->get_evaluator()->mod_switch_to_inplace(p->x_c[i].ciphertext(), data_level_1->parms_id());
			}
		}
	}else{
		p->x = new float[len];
		double sdev_inv;
		sdev_inv = 1.0 / sqrt((double)len);
		for (i = 0; i < len; ++i)
			p->x[i] = (float)(kad_drand_normal(0) * sdev_inv);
	
	}

	if (off >= 0) par[off] = p, ++(*offset);
	return p;
}

kad_node_t *kann_new_leaf2(int *offset, kad_node_p *par, uint8_t flag, float x0_01, bool is_encrypted, int n_d,  ...)
{
	int32_t i, d[KAD_MAX_DIM];
	va_list ap;
	va_start(ap, n_d); for (i = 0; i < n_d; ++i) d[i] = va_arg(ap, int); va_end(ap);
	return kann_new_leaf_array(offset, par, flag, x0_01, n_d, d, is_encrypted);
}


kad_node_t *kann_layer_bias2(int *offset, kad_node_p *par, kad_node_t *in, bool b_is_encrypted)
{
	int n0;
	kad_node_t *w, *b;
	n0 = kad_len(in);
	b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, b_is_encrypted, 1, n0);
	//return kad_cmul(kad_add(in, b), w);
	return kad_add(in, b);
}

kad_node_t *kann_layer_dense2(int *offset, kad_node_p *par, kad_node_t *in, int n1, bool w_is_encrypted, bool b_is_encrypted)
{
	int n0;
	kad_node_t *w, *b;
	n0 = kad_len(in);
	w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, w_is_encrypted, 2, n1, n0);
	b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, b_is_encrypted, 1, n1);
	//return kad_cmul(kad_add(in, b), w);
	return kad_add(kad_cmul(in, w), b);
}

kad_node_t *kann_layer_dropout2(int *offset, kad_node_p *par, kad_node_t *t, float r)
{
	kad_node_t *x[2], *cr;
	cr = kann_new_leaf2(offset, par, KAD_CONST, r, false, 0);
	x[0] = t, x[1] = kad_dropout(t, cr);
	return kad_switch(2, x);
}

/***
kad_node_t *kann_layer_layernorm2(int *offset , kad_node_t **par, kad_node_t *in)
{
	int n0;
	kad_node_t *alpha, *beta;
	n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
	alpha = kann_new_leaf2(offset, par, KAD_VAR, 1.0f, 1, n0);
	beta  = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n0);
	return kad_add(kad_mul(kad_stdnorm(in), alpha), beta);
}

static inline kad_node_t *cmul_norm2(int *offset, kad_node_t **par, kad_node_t *x, kad_node_t *w, int use_norm)
{
	return use_norm? kann_layer_layernorm2(offset, par, kad_cmul(x, w)) : kad_cmul(x, w);
}

***/

/********** APIs without offset & par **********/

kad_node_t *kann_new_leaf(uint8_t flag, float x0_01, bool is_encrypted, int n_d, ...)
{
	int32_t i, d[KAD_MAX_DIM];
	va_list ap;
	va_start(ap, n_d); for (i = 0; i < n_d; ++i) d[i] = va_arg(ap, int); va_end(ap);
	return kann_new_leaf_array(0, 0, flag, x0_01, n_d, d, is_encrypted);
}

kad_node_t *kann_new_scalar(uint8_t flag, float x, bool is_encrypted) { return kann_new_leaf(flag, x, is_encrypted, 0); }
kad_node_t *kann_new_weight(int n_row, int n_col, bool is_encrypted) { return kann_new_leaf(KAD_VAR, 0.0f, is_encrypted, 2, n_row, n_col); }
kad_node_t *kann_new_vec(int n, float x, bool is_encrypted) { return kann_new_leaf(KAD_VAR, x, is_encrypted, 1, n); }
kad_node_t *kann_new_bias(int n, bool is_encrypted) { return kann_new_vec(n, 0.0f, is_encrypted); }
kad_node_t *kann_new_weight_conv2d(int n_out, int n_in, int k_row, int k_col, bool is_encrypted) { return kann_new_leaf(KAD_VAR, 0.0f, is_encrypted, 4, n_out, n_in, k_row, k_col); }
//kad_node_t *kann_new_weight_conv1d(int n_out, int n_in, int kernel_len, bool is_encrypted) { return kann_new_leaf(KAD_VAR, 0.0f, is_encrypted, 3, n_out, n_in, kernel_len); }

kad_node_t *kann_layer_input(int n1)
{
	kad_node_t *t;
	t = kad_feed(1, n1), t->ext_flag |= KANN_F_IN;
	return t;
}
kad_node_t *kann_layer_bias(kad_node_t *in, bool b_is_encrypted) { return kann_layer_bias2(0, 0, in, b_is_encrypted); }
kad_node_t *kann_layer_dense(kad_node_t *in, int n1, bool w_is_encrypted, bool b_is_encrypted) { return kann_layer_dense2(0, 0, in, n1, w_is_encrypted, b_is_encrypted); }
kad_node_t *kann_layer_dropout(kad_node_t *t, float r) { return kann_layer_dropout2(0, 0, t, r); }
//kad_node_t *kann_layer_layernorm(kad_node_t *in) { return kann_layer_layernorm2(0, 0, in); }

/**
static kad_node_t *kann_cmul_norm(kad_node_t *x, kad_node_t *w)
{
	return kann_layer_layernorm(kad_cmul(x, w));
}
**/

kad_node_t *kann_layer_conv2d(kad_node_t *in, int n_flt, int k_rows, int k_cols, int stride_r, int stride_c, int pad_r, int pad_c, bool is_encrypted)
{
	kad_node_t *w;
	w = kann_new_weight_conv2d(n_flt, in->d[0], k_rows, k_cols, is_encrypted);
	return kad_conv2d(in, w, stride_r, stride_c, pad_r, pad_c);
}


kad_node_t *kann_layer_cost(kad_node_t *t, int n_out, int cost_type, bool is_w_encrypted, bool is_b_encrypted)
{
	kad_node_t *cost = 0, *truth = 0;
	assert(cost_type == KANN_C_CEB || cost_type == KANN_C_CEM || cost_type == KANN_C_MSE);
	t = kann_layer_dense(t, n_out, is_w_encrypted, is_b_encrypted);
	truth = kad_feed(1, n_out), truth->ext_flag |= KANN_F_TRUTH;
	if (cost_type == KANN_C_MSE) {
		cost = kad_mse(t, truth);
	} else if (cost_type == KANN_C_CEB) {
		t = kad_sigm(t);
		cost = kad_ce_bin(t, truth);
	} else if (cost_type == KANN_C_CEM) {
		t = kad_softmax(t);
		cost = kad_ce_multi(t, truth);
	}
	t->ext_flag |= KANN_F_OUT, cost->ext_flag |= KANN_F_COST;
	return cost;
}

void kann_shuffle(int n, int *s)
{
	int i, j, t;
	for (i = 0; i < n; ++i) s[i] = i;
	for (i = n; i > 0; --i) {
		j = (int)(i * kad_drand(0));
		t = s[j], s[j] = s[i-1], s[i-1] = t;
	}
}

/***************************
 *** @@MIN: minimization ***
 ***************************/

/***
 *    h0 : uniform learning rate across the model;
 *    h  : specific learning rate for different model parameters;
 *    g  : gradients
 * 	  t  : model parameters
 * 
***/
void kann_SGD(int n, float h0, const float *h, const float *g, float *t){
	int i;
	for (i = 0; i < n; ++i) {
		float lr = h? h[i] : h0;
		t[i] -= lr * g[i];
	}
}

/***
void kann_SGD(int n, float h0, const float *h, const float *g, SEALCiphertext *t){
	int i;
	for (i = 0; i < n; ++i) {
		float lr = h? h[i] : h0;
		seal_sub_inplace(t[i], lr * g[i]);
	}
}
***/


void kann_SGD(int n, float h0, const float *h, SEALCiphertext *g, SEALCiphertext *t){
	int i;
#pragma omp parallel
{
#pragma omp for
	for (i = 0; i < n; ++i) {
		int thread_mod_id = omp_get_thread_num()%omp_get_max_threads();
		float lr = h? h[i] : h0;
		if(lr != -1){
			seal_multiply(g[i], -lr, ciphertext[thread_mod_id]);
			seal_add_inplace(t[i], ciphertext[thread_mod_id]);
		}else{
			seal_add_inplace(t[i], g[i]);
		}
	}
}

}

/***
void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r)
{
	int i;
	for (i = 0; i < n; ++i) {
		float lr = h? h[i] : h0;
		r[i] = (1.0f - decay) * g[i] * g[i] + decay * r[i];
		t[i] -= lr / sqrtf(1e-6f + r[i]) * g[i];
	}
}
***/

float kann_grad_clip(float thres, int n, float *g)
{
	int i;
	double s2 = 0.0;
	for (i = 0; i < n; ++i)
		s2 += g[i] * g[i];
	s2 = sqrt(s2);
	if (s2 > thres)
		for (i = 0, s2 = 1.0 / s2; i < n; ++i)
			g[i] *= (float)s2;
	return (float)s2 / thres;
}

/****************************************************************
 *** @@XY: simpler API for network with a single input/output ***
 ****************************************************************/

//providing the truth, because this is in fact an experiment. We need to monitor the cost function & accuracy.
//Only handle the secure training part, not the data shuffling/batching/encryption part. 
//So we can count the training time.
//n : the number of *ciphertext arrays. Real total sample = n * *ciphertext.size
int kann_train_fnn1(kann_t *ann, float lr, int max_epoch, int max_drop_streak, float frac_val, int n, 
					const string& base_dir, int data_size, int label_size, vector<vector<float>>& _truth)
{
	int i, j, k,  n_train, n_val, n_in, n_out, n_var, n_encrypted_var, n_const, drop_streak = 0, min_set = 0;
	float min_val_cost = FLT_MAX, *min_x, *min_c;
	string batch_dir;

	vector<SEALCiphertext> _x(data_size); 
    vector<SEALCiphertext> _y(label_size);
	SEALCiphertext *x_ptr, *y_ptr;
	SEALCiphertext *min_x_c;

	std::chrono::high_resolution_clock::time_point train_start, train_end, load_start, load_end;
    std::chrono::microseconds training_dur(0), loading_dur(0);
	long long training_time, loading_time;

	n_in = kann_dim_in(ann);
	n_out = kann_dim_out(ann);
	if (n_in < 0 || n_out < 0) return -1;
	n_var = kann_size_var(ann);
	n_encrypted_var = kann_encrypted_var(ann);
	n_const = kann_size_const(ann);

	//instead of representing sample number, here n_train represents *ciphertext number. 
	// n_val = (int)(n * frac_val);
	n_val = 0;
	n_train = n - n_val;
	//Without plain labels, maybe we don't need to evaluate.
	// if (!_truth)
	// 	n_val = 0;
	min_x = (float*)malloc((n_var - n_encrypted_var) * sizeof(float));
	min_x_c = (SEALCiphertext*)malloc(n_encrypted_var * sizeof(SEALCiphertext));
	min_c = (float*)malloc(n_const * sizeof(float));

	for (i = 0; i < max_epoch; ++i) {
		int n_proc = 0, n_train_err = 0, n_val_err = 0, n_train_base = 0, n_val_base = 0;
		double train_cost = 0.0, val_cost = 0.0;
		int total_train_num = 0, total_val_num = 0;
		int ret_size;
		int b, c;
		kann_switch(ann, 1);
		training_time = 0;
		loading_time = 0;
		while (n_proc < n_train) {
			load_start = chrono::high_resolution_clock::now();
			batch_dir =  base_dir + "/" + to_string(n_proc);
			ret_size = load_batch_ciphertext(_x, data_size, batch_dir, 0);
			if (ret_size != data_size) {
				cout << "[train ] load data return " << ret_size << ". expect " << data_size << endl;
				goto train_exit;
			}
			ret_size = load_batch_ciphertext(_y, label_size, batch_dir, 1);
			if (ret_size != label_size) {
				cout << "[train ] load label return " << ret_size << ". expect " << label_size << endl;
				goto train_exit;
			}
			load_end = chrono::high_resolution_clock::now();
    		loading_dur = chrono::duration_cast<chrono::microseconds>(load_end - load_start);
			loading_time += loading_dur.count();

			train_start = chrono::high_resolution_clock::now();
			x_ptr = _x.data(), y_ptr = _y.data();
			kann_feed_bind(ann, KANN_F_IN,    0, &x_ptr);
			kann_feed_bind(ann, KANN_F_TRUTH, 0, &y_ptr);
			train_cost += kann_cost(ann, 0, 1);
			total_train_num += _x[0].size();
			for (k = 0; k < ann->n; k++){
				if (kad_is_var(ann->v[k])){
					if(seal_is_encrypted(ann->v[k]))
						kann_SGD(kad_len(ann->v[k]), lr, 0, ann->v[k]->g_c, ann->v[k]->x_c);
					else
						kann_SGD(kad_len(ann->v[k]), lr, 0, ann->v[k]->g, ann->v[k]->x);	
				}
			}
			// if (n_proc % 10 == 0) {
			// 	cout << "epoch: "<< i+1 << " batch: "<<n_proc<<"; training cost: " << train_cost / total_train_num;
			// 	c = kann_class_error(ann, _truth[n_proc].data(), &b);
			// 	cout << " (class error: " <<  100.0f * c / b << "%)" << endl;
			// }
			n_proc += 1;
			train_end = chrono::high_resolution_clock::now();
    		training_dur = chrono::duration_cast<chrono::microseconds>(train_end - train_start);
			training_time += training_dur.count();
		}
		cout << "Epoch " << i << " load time (us): " << loading_time
			 << " train time (us): " << training_time << endl;
		cout << "epoch: "<< i+1 << " batch: "<<n_proc<<"; training cost: " << train_cost / total_train_num;
		c = kann_class_error(ann, _truth[n_train-1].data(), &b);
		cout << " (class error: " <<  100.0f * c / b << "%)" << endl;
		train_cost /= total_train_num;
		kann_switch(ann, 0);
		//TODO: fixed evaluation set. May cause some problems when used for real tasks.
		while (n_proc < n_train + n_val) {
			batch_dir =  base_dir + "/" + to_string(n_proc);
			ret_size = load_batch_ciphertext(_x, data_size, batch_dir, 0);
			if (ret_size != data_size) {
				cout << "[eval ] load data return " << ret_size << ". expect " << data_size << endl;
				goto train_exit;
			}
			ret_size = load_batch_ciphertext(_y, label_size, batch_dir, 1);
			if (ret_size != label_size) {
				cout << "[eval ] load label return " << ret_size << ". expect " << label_size << endl;
				goto train_exit;
			}
			x_ptr = _x.data(), y_ptr = _y.data();
			kann_feed_bind(ann, KANN_F_IN,    0, &x_ptr);
			kann_feed_bind(ann, KANN_F_TRUTH, 0, &y_ptr);
			val_cost += kann_cost(ann, 0, 0);
			total_val_num += _x[0].size();
			c = kann_class_error(ann, _truth[n_proc].data(), &b);
			n_val_err += c, n_val_base += b;
			n_proc += 1;
		}
		if (n_val > 0) val_cost /= total_val_num;
		if (kann_verbose >= 3) {
			std::fprintf(stderr, "epoch: %d; training cost: %g", i+1, train_cost);
			// if (n_train_base) std::fprintf(stderr, " (class error: %.2f%%)", 100.0f * n_train_err / n_train);
			if (n_val > 0) {
				std::fprintf(stderr, "; validation cost: %g", val_cost);
				if (n_val_base) std::fprintf(stderr, " (class error: %.2f%%)", 100.0f * n_val_err / n_val_base);
			}
			fputc('\n', stderr);
		}
		if (i >= max_drop_streak && n_val > 0) {
			if (val_cost < min_val_cost) {
				min_set = 1;
				std::memcpy(min_x, ann->x, (n_var - n_encrypted_var) * sizeof(float));
				std::memcpy(min_c, ann->c, n_const * sizeof(float));
				for (k = 0; k < n_encrypted_var; k ++)min_x_c[k] = ann->x_c[k],
				drop_streak = 0;
				min_val_cost = (float)val_cost;
			} else if (++drop_streak >= max_drop_streak)
				break;
		}
	} // for
	if (min_set) {
		std::memcpy(ann->x, min_x, (n_var - n_encrypted_var) * sizeof(float));
		std::memcpy(ann->c, min_c, n_const * sizeof(float));
		for (k = 0; k < n_encrypted_var; k ++)ann->x_c[k] = min_x_c[k];
	}

train_exit:
	try{
		std::free(min_c); std::free(min_x); std::free(min_x_c); 
	}catch(exception e){
		cout<<e.what()<<endl;
	}
	return i;
}


float kann_cost_fnn1(kann_t *ann, int n, SEALCiphertext **x, SEALCiphertext **y)
{
	int n_in, n_out, n_proc = 0;
	double cost = 0.0;
	n_in = kann_dim_in(ann);
	n_out = kann_dim_out(ann);
	if (n <= 0 || n_in < 0 || n_out < 0) return 0.0;

	kann_switch(ann, 0);
	while (n_proc < n) {
		kann_feed_bind(ann, KANN_F_IN,    0, &x[n_proc]);
		kann_feed_bind(ann, KANN_F_TRUTH, 0, &y[n_proc]);
		cost += kann_cost(ann, 0, 0);
		n_proc += 1;
	}
	return (float)(cost / n);
}

SEALCiphertext *kann_apply1(kann_t *a, SEALCiphertext *x_c)
{
	int i_out;
	i_out = kann_find(a, KANN_F_OUT, 0);
	if (i_out < 0) return 0;
	kann_feed_bind(a, KANN_F_IN, 0, &x_c);
	kad_eval_at(a->n, a->v, i_out);
	return a->v[i_out]->x_c;
}

	/**
	 * Record the running times for each layer, for forward as well as backward propagation.
	 * Ps: This function presumes that ann is already runnable with bound inputs.
	 * Argument:
	 * 	repeat: int, repeat time for statistical average;
	 * 	forward_times: vector<int>, store the running times of forwarding for layers in ms;
	 * 	backward_times: vector<int>, store the running times of backwarding for layers in ms;
	 * 	vector_size: int, store the size of the generated vectors.
	 **/
void time_layer(kann_t * ann, int repeat, vector<pair<int, string>> &recorded_layers, vector<int> &forward_times, vector<int> &backward_times, vector<int> &other_times, int &fp_time, int &bp_time, int &all_time,
const string& base_dir, int data_size, int label_size){
	assert(repeat > 0);
	int i = 0;
	int ret_size = 0;
	kad_node_t ** a = ann->v;
	int n = ann->n;
	int i_cost = kann_find(ann, KANN_F_COST, 0);
	string batch_dir = base_dir + "/0";
	vector<SEALCiphertext> _x(data_size); 
    vector<SEALCiphertext> _y(label_size);
	SEALCiphertext *x_ptr, *y_ptr;

	// 1. load the data/label
	ret_size = load_batch_ciphertext(_x, data_size, batch_dir, 0);
	if (ret_size != data_size) {
		cout << "[train ] load data return " << ret_size << ". expect " << data_size << endl;
		return;
	}
	ret_size = load_batch_ciphertext(_y, label_size, batch_dir, 1);
	if (ret_size != label_size) {
		cout << "[train ] load label return " << ret_size << ". expect " << label_size << endl;
		return;
	}
	x_ptr = _x.data(), y_ptr = _y.data();
	kann_feed_bind(ann, KANN_F_IN,    0, &x_ptr);
	kann_feed_bind(ann, KANN_F_TRUTH, 0, &y_ptr);

	// 2. forwarding/backwording and recoding the timer
	accumulate_times_layer(n, a, i_cost, repeat, recorded_layers, forward_times, backward_times, other_times);

	for (i = 0; i < forward_times.size(); i++){
		forward_times[i] /= static_cast<double>(repeat);
		backward_times[i] /= static_cast<double>(repeat); 
	}
	for (i = 0; i < other_times.size(); i++){
		other_times[i] /= static_cast<double>(repeat);
	}

	// 3. calculate the forward, backward, full time.
	fp_time = 0;
	bp_time = 0;
	all_time = 0;
	for (i = 0; i < forward_times.size(); i++){
		fp_time += forward_times[i];
		bp_time += backward_times[i]; 
	}
	all_time = fp_time + bp_time;
	for (i = 0; i < other_times.size(); i++){
		all_time += other_times[i];
	}	
}