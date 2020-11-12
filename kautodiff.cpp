#include "kautodiff.h"
#include <cstdlib>
#include <cassert>
#include <cstdarg>
#include <string>
#include <vector>
#include <cfloat>
#include <cmath>
#include <iostream>

#ifndef _DEBUG // works in VS
#define DEBUG(x) 
#else
#define DEBUG(x) do { std::cerr << x << std::endl; } while (0)
#endif

bool remote = false; 
std::shared_ptr<SEALEngine> engine;
SEALPlaintext *plaintext;
SEALCiphertext *ciphertext;
vector<double> t;
vector<double> test_t;   
vector<double> truth_t;

typedef struct {
	uint64_t s[2];
	double n_gset;
	int n_iset;
	volatile int lock;
} kad_rng_t;

/**********************
 * Graph construction *
 **********************/

// Create a kad_node_t (internal, op), delay the memory allocation part.
static inline kad_node_t *kad_new_core(int n_d, int op, int n_child)
{
	kad_node_t *s;
	if (n_d >= KAD_MAX_DIM) return 0;
	s = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	s->flag |= SEAL_CIPHER; // All core nodes are encrypted in our framework.
	s->n_d = n_d, s->op = op, s->n_child = n_child;
	if (s->n_child) s->child = (kad_node_t**)calloc(s->n_child, sizeof(kad_node_t*));
	return s;
}

// Create a kad_node_t (leaf, const/variable/feed), with given memory pointers.
static inline kad_node_t *kad_vleaf(uint8_t flag, float *x, float *g, SEALCiphertext *x_c, SEALCiphertext *g_c, int n_d, va_list ap)
{
	int i;
	kad_node_t *p;
	if (n_d > KAD_MAX_DIM) return 0;
	p = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	p->n_d = n_d;
	for (i = 0; i < n_d; ++i)
		p->d[i] = va_arg(ap, int32_t);
	p->x = x, p->g = g, p->flag = flag;
    p->x_c = x_c, p->g_c = g_c;
	if(p->x_c){
        p->flag |= SEAL_CIPHER;
    }
	return p;
}

// DEPRECATED? We assume all const to be plain.
// Create a encrypted const node.
/**
kad_node_t *kad_const(SEALCiphertext *x_c, int n_d, ...){
    kad_node_t *p;
    va_list ap;
    va_start(ap, n_d); p = kad_vleaf(KAD_CONST, 0, 0, x_c, 0, n_d, ap); 
    va_end(ap);
    return p;
}
**/

// Create a plain const node.
kad_node_t *kad_const(float *x, int n_d, ...)
{
	kad_node_t *p;
	va_list ap;
	va_start(ap, n_d); p = kad_vleaf(KAD_CONST, x, 0, 0, 0, n_d, ap); va_end(ap);
	return p;
}

// Create a feed node. (must be encrypted)
kad_node_t *kad_feed(int n_d, ...)
{
	kad_node_t *p;
	va_list ap;
	va_start(ap, n_d); p = kad_vleaf(0, 0, 0, 0, 0, n_d, ap); va_end(ap);
	p->flag |= SEAL_CIPHER;
	return p;
}

// Create a var node.
// Comment: Not used in KANN... KANN uses kann_new_leaf_array to create new leaves.
kad_node_t *kad_var(SEALCiphertext *x_c, SEALCiphertext *g_c, int n_d, ...){
	kad_node_t *p;
	va_list ap;
	va_start(ap, n_d); p = kad_vleaf(KAD_VAR, 0, 0, x_c, g_c, n_d, ap); va_end(ap);
	return p;
}


// Create a var node.
// Comment: Not used in KANN... KANN uses kann_new_leaf_array to create new leaves.
kad_node_t *kad_var(float *x, float *g, int n_d, ...)
{
	kad_node_t *p;
	va_list ap;
	va_start(ap, n_d); p = kad_vleaf(KAD_VAR, x, g, 0, 0, n_d, ap); va_end(ap);
	return p;
}


// Check KAD_SYNC_DIM action (whether consistent with its children?); add flags(var, cipher) based on node information.
static inline kad_node_t *kad_finalize_node(kad_node_t *s) /* a helper function */
{
	int i;
	if (kad_op_list[s->op](s, KAD_SYNC_DIM) < 0) { /* check dimension */
		cout << "node sync dimension failed! op type: " << s->op << endl;
		if (s->ptr) std::free(s->ptr);
		std::free(s->child); std::free(s);
		return 0;
	}
	for (i = 0; i < s->n_child; ++i)
		if (kad_is_back(s->child[i]))
			break;
	if (i < s->n_child) s->flag |= KAD_VAR;
	return s;
}

/********** Simple arithmetic **********/

// Create a core node with 2 children.
static inline kad_node_t *kad_op2_core(int op, kad_node_t *x, kad_node_t *y)
{
	kad_node_t *s;
	s = kad_new_core(0, op, 2);
	s->child[0] = x, s->child[1] = y;
	return kad_finalize_node(s);
}

// Create a core node with 1 child.
static inline kad_node_t *kad_op1_core(int op, kad_node_t *x)
{
	kad_node_t *s;
	s = kad_new_core(0, op, 1);
	s->child[0] = x;
	return kad_finalize_node(s);
}

#define KAD_FUNC_OP2(fname, op) kad_node_t *fname(kad_node_t *x, kad_node_t *y) { return kad_op2_core((op), x, y); }

KAD_FUNC_OP2(kad_add, 1)
KAD_FUNC_OP2(kad_sub, 23)
KAD_FUNC_OP2(kad_mul, 2)
KAD_FUNC_OP2(kad_cmul, 3)
KAD_FUNC_OP2(kad_matmul, 9)
KAD_FUNC_OP2(kad_ce_multi, 13)
KAD_FUNC_OP2(kad_ce_bin, 22)
KAD_FUNC_OP2(kad_ce_bin_neg, 4)
KAD_FUNC_OP2(kad_mse, 29)

#define KAD_FUNC_OP1(fname, op) kad_node_t *fname(kad_node_t *x) { return kad_op1_core((op), x); }

KAD_FUNC_OP1(kad_log, 27)
KAD_FUNC_OP1(kad_exp, 33)
KAD_FUNC_OP1(kad_sin, 34)
KAD_FUNC_OP1(kad_square, 5)
KAD_FUNC_OP1(kad_sigm, 6)
KAD_FUNC_OP1(kad_tanh, 7)
KAD_FUNC_OP1(kad_relu, 8)
KAD_FUNC_OP1(kad_1minus, 11)
KAD_FUNC_OP1(kad_softmax, 14)
KAD_FUNC_OP1(kad_stdnorm, 32)


kad_node_t *kad_ce_multi_weighted(kad_node_t *pred, kad_node_t *truth, kad_node_t *weight)
{
	kad_node_t *s;
	s = kad_new_core(0, 13, 3);
	s->child[0] = pred, s->child[1] = truth, s->child[2] = weight;
	return kad_finalize_node(s);
}

/********** Convolution **********/

/* compute output dimension and padding sizes on both sides */
// what is pad0?
static inline int conv_find_par(int in_size, int kernel_size, int stride, int pad0, int *new_pad0, int *new_pad1)
{
	int out_size, pad_both;
	/* key equation: out_size = (in_size - kernel_size + pad_both) / stride + 1 */
	if (pad0 == KAD_PAD_SAME && stride == 1) out_size = in_size;
	else out_size = (in_size - kernel_size + (pad0 > 0? pad0 : 0) + stride - 1) / stride + 1;
	pad_both = (out_size - 1) * stride + kernel_size - in_size;
	*new_pad0 = pad_both / 2;
	*new_pad1 = pad_both - *new_pad0;
	return out_size;
}

typedef struct {
	int kernel_size, stride, pad[2];
} conv_conf_t;


// create cnn related information: strides, paddings and kernel sizes.
static inline conv_conf_t *conv2d_gen_aux(int in_row, int in_col, int kernel_r, int kernel_c, int stride_r, int stride_c, int top_pad, int left_pad)
{
	conv_conf_t *cnn;
	cnn = (conv_conf_t*)calloc(2, sizeof(conv_conf_t));
	cnn[0].kernel_size = kernel_r, cnn[0].stride = stride_r;
	cnn[1].kernel_size = kernel_c, cnn[1].stride = stride_c;
	conv_find_par(in_row, kernel_r, stride_r, top_pad,  &cnn[0].pad[0], &cnn[0].pad[1]);
	conv_find_par(in_col, kernel_c, stride_c, left_pad, &cnn[1].pad[0], &cnn[1].pad[1]);
	return cnn;
}

kad_node_t *kad_conv2d(kad_node_t *x, kad_node_t *w, int stride_r, int stride_c, int top_pad, int left_pad)
{
	kad_node_t *s;
	if (x->n_d != 3 || w->n_d != 4) return 0;
	s = kad_new_core(0, 16, 2);
	s->child[0] = x, s->child[1] = w;
	s->ptr = conv2d_gen_aux(x->d[1], x->d[2], w->d[2], w->d[3], stride_r, stride_c, top_pad, left_pad);
	s->ptr_size = sizeof(conv_conf_t) * 2;
	return kad_finalize_node(s);
}

kad_node_t *kad_max2d(kad_node_t *x, int kernel_r, int kernel_c, int stride_r, int stride_c, int top_pad, int left_pad)
{
	kad_node_t *s;
	if (x->n_d != 3) return 0;
	s = kad_new_core(0, 17, 1);
	s->child[0] = x;
	s->ptr = conv2d_gen_aux(x->d[1], x->d[2], kernel_r, kernel_c, stride_r, stride_c, top_pad, left_pad);
	s->ptr_size = sizeof(conv_conf_t) * 2;
	return kad_finalize_node(s);
}

/********** Multi-node pooling **********/

static kad_node_t *kad_pooling_general(int op, int n, kad_node_t **x)
{
	int i;
	kad_node_t *s;
	s = kad_new_core(0, op, n);
	s->flag |= KAD_POOL;
	for (i = 0; i < n; ++i)
		s->child[i] = x[i];
	return kad_finalize_node(s);
}

kad_node_t *kad_avg(int n, kad_node_t **x)   { return kad_pooling_general(10, n, x); }
kad_node_t *kad_max(int n, kad_node_t **x)   { return kad_pooling_general(21, n, x); }
kad_node_t *kad_stack(int n, kad_node_t **x) { return kad_pooling_general(35, n, x); }

kad_node_t *kad_select(int n, kad_node_t **x, int which)
{
	kad_node_t *s;
	int32_t i, *aux;
	aux = (int32_t*)calloc(1, 4);
	*aux = which;
	s = kad_new_core(0, 12, n);
	for (i = 0; i < n; ++i) s->child[i] = x[i];
	s->flag |= KAD_POOL, s->ptr = aux, s->ptr_size = 4;
	return kad_finalize_node(s);
}

/********** Dimension reduction **********/
/**
static kad_node_t *kad_reduce_general(int op, kad_node_t *x, int axis)
{
	kad_node_t *s;
	int32_t *aux;
	aux = (int32_t*)malloc(4);
	aux[0] = axis;
	s = kad_new_core(0, op, 1);
	s->child[0] = x;
	s->ptr = aux, s->ptr_size = 4;
	return kad_finalize_node(s);
}

kad_node_t *kad_reduce_sum(kad_node_t *x, int axis)  { return kad_reduce_general(25, x, axis); }
kad_node_t *kad_reduce_mean(kad_node_t *x, int axis) { return kad_reduce_general(26, x, axis); }
**/
/********** Sampling related **********/

kad_node_t *kad_dropout(kad_node_t *x, kad_node_t *y)
{
	kad_node_t *z;
	z = kad_op2_core(15, x, y);
	z->ptr = kad_rng(), z->ptr_size = sizeof(kad_rng_t);
	return z;
}

kad_node_t *kad_sample_normal(kad_node_t *x)
{
	kad_node_t *z;
	z = kad_op1_core(24, x);
	z->ptr = kad_rng(), z->ptr_size = sizeof(kad_rng_t);
	return z;
}

/********** Miscellaneous **********/
/**
kad_node_t *kad_slice(kad_node_t *x, int axis, int start, int end)
{
	kad_node_t *s;
	int32_t *aux;
	if (end < start || start < 0) return 0;
	aux = (int32_t*)malloc(3 * 4);
	aux[0] = axis, aux[1] = start, aux[2] = end;
	s = kad_new_core(0, 20, 1);
	s->child[0] = x;
	s->ptr = aux, s->ptr_size = 3 * 4;
	return kad_finalize_node(s);
}

kad_node_t *kad_concat_array(int axis, int n, kad_node_t **p)
{
	kad_node_t *s;
	int32_t i, *aux;
	aux = (int32_t*)malloc(4);
	aux[0] = axis;
	s = kad_new_core(0, 31, n);
	for (i = 0; i < n; ++i)
		s->child[i] = p[i];
	s->ptr = aux, s->ptr_size = 4;
	return kad_finalize_node(s);
}

kad_node_t *kad_concat(int axis, int n, ...)
{
	int i;
	kad_node_t **p, *s;
	va_list ap;
	p = (kad_node_t**)malloc(n * sizeof(kad_node_t*));
	va_start(ap, n);
	for (i = 0; i < n; ++i) p[i] = va_arg(ap, kad_node_p);
	va_end(ap);
	s = kad_concat_array(axis, n, p);
	std::free(p);
	return s;
}

kad_node_t *kad_reshape(kad_node_t *x, int n_d, int *d)
{
	kad_node_t *s;
	int32_t i, *aux = 0;
	if (n_d > 0) {
		aux = (int32_t*)malloc(n_d * 4);
		for (i = 0; i < n_d; ++i) aux[i] = d? d[i] : -1;
	}
	s = kad_new_core(0, 30, 1);
	s->child[0] = x, s->ptr = aux, s->ptr_size = n_d * 4;
	return kad_finalize_node(s);
}

kad_node_t *kad_reverse(kad_node_t *x, int axis)
{
	kad_node_t *s;
	int32_t *aux;
	aux = (int32_t*)malloc(4);
	*aux = axis;
	s = kad_new_core(0, 36, 1);
	s->child[0] = x, s->ptr = aux, s->ptr_size = 4;
	return kad_finalize_node(s);
}
**/
kad_node_t *kad_switch(int n, kad_node_t **p)
{
	kad_node_t *s;
	int32_t i, *aux;
	aux = (int32_t*)calloc(1, 4);
	s = kad_new_core(0, 12, n);
	for (i = 0; i < n; ++i)
		s->child[i] = p[i];
	s->ptr = aux, s->ptr_size = 4;
	return kad_finalize_node(s);
}

/***********************
 * Graph linearization *
 ***********************/

static void kad_mark_back(int n, kad_node_t **v)
{
	int i, j;
	for (i = 0; i < n; ++i) {
		if (v[i]->n_child == 0) continue;
		for (j = 0; j < v[i]->n_child; ++j)
			if (kad_is_back(v[i]->child[j]))
				break;
		if (j < v[i]->n_child) v[i]->flag |= KAD_VAR;
		else v[i]->flag &= ~KAD_VAR;
	}
}

//think carefully about the reallocation: we cannot afford calling it often.
// allocate internal nodes memory.
// Assumption: All internal nodes are encrypted.
static void kad_allocate_internal(int n, kad_node_t **v)
{
	int i, j;
	kad_mark_back(n, v);
	for (i = 0; i < n; ++i) {
		kad_node_t *p = v[i];
		if (p->n_child == 0) continue;
		if(p->x_c) delete[] p->x_c;
        p->x_c = new SEALCiphertext[kad_len(p)];
		for(j = 0; j< kad_len(p); j++)
			p->x_c[j].init(engine);
        if (kad_is_back(p)) {
			if(p->g_c) delete[] p->g_c;
			p->g_c = new SEALCiphertext[kad_len(p)];
			for(j = 0; j< kad_len(p); j++)
				p->g_c[j].init(engine);
			if(p->g) delete[] p->g;	
			p->g = new float[kad_len(p)];
            kad_op_list[p->op](p, KAD_ALLOC);
        }
	}
}
    
#define kvec_t(type) struct { size_t n, m; type *a; }

#define kv_pop(v) ((v).a[--(v).n])

#define kv_push(type, v, x) do { \
		if ((v).n == (v).m) { \
			(v).m = (v).m? (v).m<<1 : 2; \
			(v).a = (type*)realloc((v).a, sizeof(type) * (v).m); \
		} \
		(v).a[(v).n++] = (x); \
	} while (0)

/* IMPORTANT: kad_node_t::tmp MUST BE set to zero before calling this function */
// topological sorting nodes, given multiple tree roots, list the nodes including leaves and internal nodes all into a list, 
// then allocate memory for internal nodes.
kad_node_t **kad_compile_array(int *n_node, int n_roots, kad_node_t **roots)
{
	int i;
	kvec_t(kad_node_p) stack = {0,0,0}, a = {0,0,0};

	/* generate kad_node_t::tmp, the count of the parent nodes; shifted by 1; lowest bit to detect fake roots */
	for (i = 0; i < n_roots; ++i) {
		roots[i]->tmp = 1; /* mark the root */ 
		kv_push(kad_node_p, stack, roots[i]);
	}
	while (stack.n) {
		kad_node_t *p = kv_pop(stack);
		for (i = 0; i < p->n_child; ++i) {
			kad_node_t *q = p->child[i];
			if (q->tmp == 0) kv_push(kad_node_p, stack, q);
			q->tmp += 1<<1;
		}
	}

	/* topological sorting (Kahn's algorithm) */
	for (i = 0; i < n_roots; ++i)
		if (roots[i]->tmp>>1 == 0) /* if roots[i]->tmp>>1 != 0, it is not a real root */
			kv_push(kad_node_p, stack, roots[i]);
	while (stack.n) {
		kad_node_t *p = kv_pop(stack);
		kv_push(kad_node_p, a, p);
		for (i = 0; i < p->n_child; ++i) {
			p->child[i]->tmp -= 1<<1;
			if (p->child[i]->tmp>>1 == 0)
				kv_push(kad_node_p, stack, p->child[i]);
		}
	}
	std::free(stack.a);
	for (i = 0; i < (int)a.n; ++i) { /* check cycles; no cycles if constructed with kad_add() etc */
		assert(a.a[i]->tmp>>1 == 0);
		a.a[i]->tmp = 0;
	}

	/* reverse */
	for (i = 0; i < (int)a.n>>1; ++i) { /* reverse a.a[] */
		kad_node_p t;
		t = a.a[i], a.a[i] = a.a[a.n-1-i], a.a[a.n-1-i] = t;
	}
	kad_allocate_internal(a.n, a.a);

	*n_node = a.n;
	return a.a;
}

// kad_compile: from variange roots to root list, then call kad_compile_array.
kad_node_t **kad_compile(int *n_node, int n_roots, ...)
{
	int i;
	kad_node_t **roots, **ret;
	va_list ap;

	roots = (kad_node_t**)malloc(n_roots * sizeof(kad_node_t*));
	va_start(ap, n_roots);
	for (i = 0; i < n_roots; ++i) roots[i] = va_arg(ap, kad_node_p);
	va_end(ap);
	ret = kad_compile_array(n_node, n_roots, roots);
	std::free(roots);
	return ret;
}

/************************************
 * Miscellaneous on compiled graphs *
 ************************************/

// Only free internal memory.
void kad_delete(int n, kad_node_t **a)
{
	int i;
	for (i = 0; i < n; ++i) {
		kad_node_t *p = a[i];
		if (p->n_child) {
			// All internal nodes are encrypted.
			delete[] p->x_c, p->g_c, p->g; 
		}
		std::free(p->child); 
		std::free(p->ptr); 
		std::free(p->gtmp); 
		std::free(p);
	}
	std::free(a);
}

int kad_size_var(int n, kad_node_t *const* v)
{
	int c, i;
	for (i = c = 0; i < n; ++i)
		if (kad_is_var(v[i]))
			c += kad_len(v[i]);
	return c;
}


int kad_size_encrypted_var(int n, kad_node_t *const* v){
	int c, i;
	for (i = c = 0; i < n; ++i)
		if (kad_is_var(v[i]) && seal_is_encrypted(v[i]))
			c += kad_len(v[i]);
	return c;
}



int kad_size_const(int n, kad_node_t *const* v)
{
	int c, i;
	for (i = c = 0; i < n; ++i)
		if (kad_is_const(v[i]))
			c += kad_len(v[i]);
	return c;
}

/**********************************
 * Computate values and gradients *
 **********************************/

// Propagate marks, have to add a initial mask before calling it.
static void kad_propagate_marks(int n, kad_node_t **a)
{
	int i, j;
	for (i = n - 1; i >= 0; --i) {
		kad_node_t *p = a[i];
		if (p->tmp > 0) {
			if (kad_is_switch(p)) {
				int32_t *aux = (int32_t*)p->ptr;
				if (p->child[*aux]->tmp == 0)
					p->child[*aux]->tmp = 1;
			} else {
				for (j = 0; j < p->n_child; ++j)
					if (p->child[j]->tmp == 0)
						p->child[j]->tmp = 1;
			}
		}
	}
}

// Execute forward operation on masked nodes.
void kad_eval_marked(int n, kad_node_t **a)
{
	int i;
	kad_propagate_marks(n, a);
	for (i = 0; i < n; ++i)
		if (a[i]->n_child && a[i]->tmp > 0){
			cout << "forwarding node :" << i << " op " << kad_op_name[a[i]->op] << endl;
			kad_op_list[a[i]->op](a[i], KAD_FORWARD);
			//kad_op_cmul(a[i], KAD_FORWARD, a);
		}
	for (i = 0; i < n; ++i) a[i]->tmp = 0;
}

// mask the 'from' node, then call kad_eval_marked.
const SEALCiphertext *kad_eval_at(int n, kad_node_t **a, int from)
{
	int i;
	if(!seal_is_encrypted(a[from])){
		throw invalid_argument("Only evaluate ciphertext for now.");
	}
	if (from < 0 || from >= n) from = n - 1;
	for (i = 0; i < n; ++i) a[i]->tmp = (i == from);
	kad_eval_marked(n, a);
	return a[from]->x_c;
}

// BP from the 'from' node, where the initial gradient resides.
void kad_grad(int n, kad_node_t **a, int from, bool add_noise)
{
	int i, j;
	if (from < 0 || from >= n) from = n - 1;
	assert(a[from]->n_d == 0); // the 'from' node should be a scalar node.
	for (i = 0; i < n; ++i) a[i]->tmp = (i == from);
	kad_propagate_marks(n, a);
	for (i = 0; i <= from; ++i) /* set all grandients to zero */{
        if (a[i]->tmp >0){ //feed doesn't have gradient.
			if ((a[i]->g))
				memset(a[i]->g, 0, kad_len(a[i]) * sizeof(float));
			if ((a[i]->g_c))
            	for(j = 0; j < kad_len(a[i]); j++) a[i]->g_c[j].clean() = true;
        }
    }
	for (i = from, a[i]->g[0] = 1; i >= 0; --i) /* backprop */
		if (a[i]->n_child && a[i]->tmp > 0){
			cout << "back propagating :" << i << " op " << kad_op_name[a[i]->op] << endl;
			kad_op_list[a[i]->op](a[i], KAD_BACKWARD);
		}
	//sum encrypted gradients
	for (i = 0; i <= from; ++i){
		if (kad_is_var(a[i]) && a[i]->g_c && a[i]->tmp > 0){
			if (!(seal_is_encrypted(a[i]))){
				if(remote){
					assert(false);
				}else{
					for(j = 0; j < kad_len(a[i]); j++){
						if (a[i]->g_c[j].clean()) {
							a[i]->g[j] = 0.0;
						}else{
							engine->decrypt(a[i]->g_c[j], *plaintext);
							engine->decode(*plaintext, t);
							//here is where noise should be added.
							a[i]->g[j] = std::accumulate(t.begin(), t.end(), 0);		
						}
					}
				}
			}
			else{
				for(j = 0; j < kad_len(a[i]); j++){
					if (a[i]->g_c[j].clean()){
						continue;
					}else{
						// maybe faster on the client side to decrypt+sum+encrypte?
						hewrapper::sum_vector(a[i]->g_c[j]);
					}

				}

			}
		}
	}
	for (i = 0; i <= from; ++i) a[i]->tmp = 0;

}

/***********************
 * Load and save graph *
 ***********************/

/**
static void kad_save1(FILE *fp, const kad_node_t *p)
{
	fwrite(&p->ext_label, 4, 1, fp);
	fwrite(&p->ext_flag, 4, 1, fp);
	fwrite(&p->flag, 1, 1, fp);
    fwrite(&p->grad_flag, 1, 1, fp);
	fwrite(&p->n_child, 4, 1, fp);
	if (p->n_child) {
		int32_t j, pre = p->pre? p->pre->tmp : -1;
		fwrite(&p->op, 2, 1, fp);
		for (j = 0; j < p->n_child; ++j)
			fwrite(&p->child[j]->tmp, 4, 1, fp);
		fwrite(&pre, 4, 1, fp);
		fwrite(&p->ptr_size, 4, 1, fp);
		if (p->ptr_size > 0 && p->ptr)
			fwrite(p->ptr, p->ptr_size, 1, fp);
	} else {
		fwrite(&p->n_d, 1, 1, fp);
		if (p->n_d) fwrite(p->d, 4, p->n_d, fp);
	}
}


static kad_node_t *kad_load1(FILE *fp, kad_node_t **node)
{
	kad_node_t *p;
	p = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	fread(&p->ext_label, 4, 1, fp);
	fread(&p->ext_flag, 4, 1, fp);
	fread(&p->flag, 1, 1, fp);
    fread(&p->grad_flag, 1, 1, fp);
	fread(&p->n_child, 4, 1, fp);
	if (p->n_child) {
		int32_t j, k;
		p->child = (kad_node_t**)calloc(p->n_child, sizeof(kad_node_t*));
		fread(&p->op, 2, 1, fp);
		for (j = 0; j < p->n_child; ++j) {
			fread(&k, 4, 1, fp);
			p->child[j] = node? node[k] : 0;
		}
		fread(&k, 4, 1, fp);
		if (k >= 0) p->pre = node[k];
		fread(&p->ptr_size, 4, 1, fp);
		if (p->ptr_size > 0) {
			p->ptr = malloc(p->ptr_size);
			fread(p->ptr, p->ptr_size, 1, fp);
		}
	} else {
		fread(&p->n_d, 1, 1, fp);
		if (p->n_d) fread(p->d, 4, p->n_d, fp);
	}
	return p;
}

int kad_save(FILE *fp, int n_node, kad_node_t **node)
{
	int32_t i, k = n_node;
	fwrite(&k, 4, 1, fp);
	for (i = 0; i < n_node; ++i) node[i]->tmp = i;
	for (i = 0; i < n_node; ++i) kad_save1(fp, node[i]);
	for (i = 0; i < n_node; ++i) node[i]->tmp = 0;
	return 0;
}

kad_node_t **kad_load(FILE *fp, int *_n_node)
{
	int32_t i, n_node;
	kad_node_t **node;
	fread(&n_node, 4, 1, fp);
	node = (kad_node_t**)malloc(n_node * sizeof(kad_node_t*));
	for (i = 0; i < n_node; ++i) {
		kad_node_t *p;
		p = node[i] = kad_load1(fp, node);
		if (p->n_child) {//only nodes with child have op.
			kad_op_list[p->op](p, KAD_ALLOC);
			kad_op_list[p->op](p, KAD_SYNC_DIM);
		}
	}
	*_n_node = n_node;
	kad_mark_back(n_node, node);
	return node;
}
**/


/***************
 * Graph clone *
 ***************/
#if 0
static inline kad_node_t *kad_dup1(const kad_node_t *p)
{
	kad_node_t *q;
	q = (kad_node_t*)malloc(sizeof(kad_node_t));
	std::memcpy(q, p, sizeof(kad_node_t));
	q->pre = 0, q->tmp = 0, q->gtmp = 0;
	if (p->ptr && p->ptr_size > 0) {
		if (kad_use_rng(p) && !(p->flag & KAD_SHARE_RNG) && p->ptr_size == sizeof(kad_rng_t)) {
			q->ptr = kad_rng(); /* each time step uses a different RNG */
		} else {
			q->ptr = malloc(p->ptr_size);
			std::memcpy(q->ptr, p->ptr, p->ptr_size);
		}
	}
    //reuse the const/init var/feed's x/g
	if (q->n_child) {
		q->x = q->g = q->x_c = q->g_c = 0
		q->child = (kad_node_t**)calloc(q->n_child, sizeof(kad_node_t*));
	}
	return q;
}

kad_node_t **kad_clone(int n, kad_node_t **v, int batch_size)
{
	int i, j, k;
	kad_node_t **u;
	u = (kad_node_t**)calloc(n, sizeof(kad_node_t*));
	for (i = 0; i < n; ++i) v[i]->tmp = i;
	for (i = 0; i < n; ++i) {
		kad_node_t *p = v[i], *q;
		q = u[i] = kad_dup1(p);
		if (p->pre) q->pre = u[p->pre->tmp];
		if (p->n_child) {
			for (j = 0; j < p->n_child; ++j)
				q->child[j] = u[p->child[j]->tmp];
		} else if (!kad_is_feed(p)) {
            if(seal_is_encrypted(p)){
                q->x_c = (SEALCiphertext*)malloc(kad_len(p) * sizeof(SEALCiphertext));
                std::memcpy(q->x_c, p->x_c, kad_len(p) * sizeof(SEALCiphertext));
            }else{
                q->x = (float*)malloc(kad_len(p) * sizeof(float));
                std::memcpy(q->x, p->x, kad_len(p) * sizeof(float));
            }
            q->g = q->g_c = 0;
		}
	}
	for (i = 0; i < n; ++i) v[i]->tmp = 0;
    //init external var above, init internal var using this function.
    kad_allocate_internal(n, v);
	return u;
}
#endif
/********************************
 * Vector and matrix operations *
 ********************************/

static inline SEALCiphertext kad_sdot(int n, SEALCiphertext *x, SEALCiphertext *y){
    int i;
    SEALCiphertext s(engine);
    s.clean() = true;
    for (i = 0; i < n; i++) {
        seal_multiply(x[i], y[i], *ciphertext);
        seal_add_inplace(s, *ciphertext);
    }
    return s;
}

static inline SEALCiphertext kad_sdot(int n, SEALCiphertext *x, const float *y){
    int i;
    SEALCiphertext s(engine);
    s.clean() = true;
    for (i = 0; i < n; i++) {
        seal_multiply(x[i], y[i], *ciphertext);
        seal_add_inplace(s, *ciphertext);
    }
    return s;
}

static inline float kad_sdot(int n, const float *x, const float *y) /* BLAS sdot */
{
	int i;
	float s = 0.;
	for (i = 0; i < n; ++i) s += x[i] * y[i];
	return s;
}


static inline void kad_saxpy_inlined(int n, SEALCiphertext a, SEALCiphertext *x, SEALCiphertext *y){
    int i;
    for (i = 0; i < n; ++i){
        seal_multiply(x[i], a, *ciphertext);
        seal_add_inplace(y[i], *ciphertext);
    }
}

static inline void kad_saxpy_inlined(int n, float a, SEALCiphertext *x, SEALCiphertext *y){
    int i;
    for (i = 0; i <n; ++i){
        seal_multiply(x[i], a, *ciphertext);
        seal_add_inplace(y[i], *ciphertext);
    }
}

static inline void kad_saxpy_inlined(int n, float a, const float *x, SEALCiphertext *y){
    int i;
    for (i = 0; i <n; ++i){
        seal_add_inplace(y[i], (x[i]*a));
    }
}


static inline void kad_saxpy_inlined(int n, SEALCiphertext a, const float *x, SEALCiphertext *y){
    int i;
    for (i = 0; i < n; ++i){
        seal_multiply(a, x[i], *ciphertext);
        seal_add_inplace(y[i], *ciphertext);
    }
}

static inline void kad_saxpy_inlined(int n, float a, const float *x, float *y) // BLAS saxpy
{
	int i;
	for (i = 0; i < n; ++i) y[i] += a * x[i];
}

template<typename T>
void kad_vec_mul_sum(int n, SEALCiphertext *a, SEALCiphertext *b, T *c){
	static_assert(std::is_same<T, SEALCiphertext>::value || std::is_same<T, SEALPlaintext>::value || std::is_same<T, float>::value,"Bad T");
    int i;
    for (i = 0; i < n; ++i){
        seal_multiply(b[i], c[i], *ciphertext);
        seal_add_inplace(a[i], *ciphertext);
    }
}

void kad_vec_mul_sum(int n, float *a, const float *b, const float *c)
{
	int i;
	for (i = 0; i < n; ++i) a[i] += b[i] * c[i];
}

template<typename A, typename B, typename C>
void kad_saxpy(int n, B a, A *x, C *y){
    kad_saxpy_inlined(n, a, x, y);
}


template<typename T> 
void kad_sgemm_simple(int trans_A, int trans_B, int M, int N, int K, SEALCiphertext *A, T *B, SEALCiphertext *C){
    static const int x = 16;
	int i, j, k;
	if (!trans_A && trans_B) {
		for (i = 0; i < M; i += x)
			for (j = 0; j < N; j += x) {
				int ii, ie = M < i + x? M : i + x;
				int jj, je = N < j + x? N : j + x;
				for (ii = i; ii < ie; ++ii) { /* loop tiling */
					SEALCiphertext *aii = A + ii * K;
                    T  *bjj;
					SEALCiphertext *cii = C + ii * N;
					for (jj = j, bjj = B + j * K; jj < je; ++jj, bjj += K){
                        *ciphertext = kad_sdot(K, aii, bjj);
						seal_add_inplace(cii[jj], *ciphertext);
                    }
				}
			}
	} else if (!trans_A && !trans_B) {
		for (i = 0; i < M; ++i)
			for (k = 0; k < K; ++k)
				kad_saxpy_inlined(N, A[i*K+k], &B[k*N], &C[i*N]); 
	} else if (trans_A && !trans_B) {
		for (k = 0; k < K; ++k)
			for (i = 0; i < M; ++i)
				kad_saxpy_inlined(N, A[k*M+i], &B[k*N], &C[i*N]);
	} else abort(); /* not implemented for (trans_A && trans_B) */
}

/***************************
 * Random number generator *
 ***************************/

static kad_rng_t kad_rng_dat = { {0x50f5647d2380309dULL, 0x91ffa96fc4c62cceULL}, 0.0, 0, 0 };

static inline uint64_t kad_splitmix64(uint64_t x)
{
	uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static inline uint64_t kad_xoroshiro128plus_next(kad_rng_t *r)
{
	const uint64_t s0 = r->s[0];
	uint64_t s1 = r->s[1];
	const uint64_t result = s0 + s1;
	s1 ^= s0;
	r->s[0] = (s0 << 55 | s0 >> 9) ^ s1 ^ (s1 << 14);
	r->s[1] = s0 << 36 | s0 >> 28;
	return result;
}

static inline void kad_xoroshiro128plus_jump(kad_rng_t *r)
{
	static const uint64_t JUMP[] = { 0xbeac0467eba5facbULL, 0xd86b048b86aa9922ULL };
	uint64_t s0 = 0, s1 = 0;
	int i, b;
	for (i = 0; i < 2; ++i)
		for (b = 0; b < 64; b++) {
			if (JUMP[i] & 1ULL << b)
				s0 ^= r->s[0], s1 ^= r->s[1];
			kad_xoroshiro128plus_next(r);
		}
	r->s[0] = s0, r->s[1] = s1;
}

void kad_srand(void *d, uint64_t seed)
{
	kad_rng_t *r = d? (kad_rng_t*)d : &kad_rng_dat;
	r->n_gset = 0.0, r->n_iset = 0;
	r->s[0] = kad_splitmix64(seed);
	r->s[1] = kad_splitmix64(r->s[0]);
}

void *kad_rng(void)
{
	kad_rng_t *r;
	r = (kad_rng_t*)calloc(1, sizeof(kad_rng_t));
	kad_xoroshiro128plus_jump(&kad_rng_dat);
	r->s[0] = kad_rng_dat.s[0], r->s[1] = kad_rng_dat.s[1];
	return r;
}

uint64_t kad_rand(void *d) { return kad_xoroshiro128plus_next(d? (kad_rng_t*)d : &kad_rng_dat); }

double kad_drand(void *d)
{
	union { uint64_t i; double d; } u;
	u.i = 0x3FFULL << 52 | kad_xoroshiro128plus_next(d? (kad_rng_t*)d : &kad_rng_dat) >> 12;
	return u.d - 1.0;
}

double kad_drand_normal(void *d)
{
	kad_rng_t *r = d? (kad_rng_t*)d : &kad_rng_dat;
	if (r->n_iset == 0) {
		double fac, rsq, v1, v2;
		do {
			v1 = 2.0 * kad_drand(d) - 1.0;
			v2 = 2.0 * kad_drand(d) - 1.0;
			rsq = v1 * v1 + v2 * v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac = sqrt(-2.0 * log(rsq) / rsq);
		r->n_gset = v1 * fac;
		r->n_iset = 1;
		return v2 * fac;
	} else {
		r->n_iset = 0;
		return r->n_gset;
	}
}

/*************
 * Operators *
 *************/

static inline void kad_copy_dim1(kad_node_t *dst, const kad_node_t *src) /* set the dimension/shape of dst to src */
{
	dst->n_d = src->n_d;
	if (src->n_d) std::memcpy(dst->d, src->d, src->n_d * sizeof(int));
}

/********** Arithmetic operations **********/

int kad_op_add(kad_node_t *p, int action)
{
	int i, n0, n1;
	kad_node_t *q[2];
	q[0] = p->child[0], n0 = kad_len(q[0]);
	q[1] = p->child[1], n1 = kad_len(q[1]);
	if (action == KAD_SYNC_DIM) {
		if (n0 % n1 != 0) return -1;
		kad_copy_dim1(p, q[0]);
	} else if (action == KAD_FORWARD) {
		assert(n0 >= n1);
		for (i = 0; i < n0; i++)
			p->x_c[i] = q[0]->x_c[i];
        if (seal_is_encrypted(q[1]))
            for (i = 0; i < n0; i += n1)
                kad_saxpy(n1, 1.0f, q[1]->x_c, p->x_c + i);
        else
            for (i = 0; i < n0; i += n1)
                kad_saxpy(n1, 1.0f, q[1]->x, p->x_c + i);
	} else if (action == KAD_BACKWARD) {
        if (kad_is_back(q[0]))
            kad_saxpy(n0, 1.0f, p->g_c, q[0]->g_c);
        if (kad_is_back(q[1]))
            for (i = 0; i < n0; i += n1)
                kad_saxpy(n1, 1.0f, p->g_c + i, q[1]->g_c);
    }
	return 0;
}

int kad_op_sub(kad_node_t *p, int action)
{
	int i, n0, n1;
	kad_node_t *q[2];

	q[0] = p->child[0], n0 = kad_len(q[0]);
	q[1] = p->child[1], n1 = kad_len(q[1]);
	if (action == KAD_SYNC_DIM) {
		if (n0 % n1 != 0) return -1;
		kad_copy_dim1(p, q[0]);
	} else if (action == KAD_FORWARD) {
		assert(n0 >= n1);
		for (i = 0; i < n0; i++)
			p->x_c[i] = q[0]->x_c[i];
        if (seal_is_encrypted(q[1]))
            for (i = 0; i < n0; i += n1)
                kad_saxpy(n1, -1.0f, q[1]->x_c, p->x_c + i);
        else
            for (i = 0; i < n0; i += n1)
                kad_saxpy(n1, -1.0f, q[1]->x, p->x + i);
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q[0])) kad_saxpy(n0, 1.0f, p->g_c, q[0]->g_c);
		if (kad_is_back(q[1]))
			for (i = 0; i < n0; i += n1)
				kad_saxpy(n1, -1.0f, p->g_c + i, q[1]->g_c);
	}
	return 0;
}

int kad_op_mul(kad_node_t *p, int action)
{
	int i, j, n0, n1;
	kad_node_t *q[2];

	q[0] = p->child[0], n0 = kad_len(q[0]);
	q[1] = p->child[1], n1 = kad_len(q[1]);
	if (action == KAD_SYNC_DIM) {
		if (n0 % n1 != 0) return -1;
		kad_copy_dim1(p, q[0]);
	} else if (action == KAD_FORWARD) {
		assert(n0 >= n1);
        for(j = 0; j < kad_len(p); j++)
            p->x_c[j].clean() = true;
        if (seal_is_encrypted(q[1]))
            for (i = 0; i < n0; i += n1) /* TODO: optimize when n1==1 */
                kad_vec_mul_sum(n1, p->x_c + i, q[0]->x_c + i, q[1]->x_c);
        else
            for (i = 0; i < n0; i += n1) /* TODO: optimize when n1==1 */
                kad_vec_mul_sum(n1, p->x_c + i, q[0]->x_c + i, q[1]->x);
		//if (q[0]->x != 0 && q[1]->x != 0) //why it's possible? comment it for now.
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q[0]))
            if (seal_is_encrypted(q[1]))
                for (i = 0; i < n0; i += n1)
                    kad_vec_mul_sum(n1, q[0]->g_c + i, p->g_c + i, q[1]->x_c);
            else
                for (i = 0; i < n0; i += n1)
                    kad_vec_mul_sum(n1, q[0]->g_c + i, p->g_c + i, q[1]->x);
		if (kad_is_back(q[1]))
			for (i = 0; i < n0; i += n1)
				kad_vec_mul_sum(n1, q[1]->g_c, p->g_c + i, q[0]->x_c + i);
	}
	return 0;
}


// column (last dimension) inner product.
int kad_op_cmul(kad_node_t *p, int action)
{
	int i, n_a_row, n_b_row, n_col, n_a_col = 1, n_b_col = 1;
    int j;
	kad_node_t *q[2];

	q[0] = p->child[0], q[1] = p->child[1];
	n_col = q[0]->d[q[0]->n_d - 1] > q[1]->d[q[1]->n_d - 1]? q[0]->d[q[0]->n_d - 1] : q[1]->d[q[1]->n_d - 1];
	for (i = q[0]->n_d - 1; i >= 0; --i) if (n_a_col < n_col) n_a_col *= q[0]->d[i];
	for (i = q[1]->n_d - 1; i >= 0; --i) if (n_b_col < n_col) n_b_col *= q[1]->d[i];
	n_a_row = kad_len(q[0]) / n_a_col, n_b_row = kad_len(q[1]) / n_b_col;
	if (action == KAD_SYNC_DIM) {
		if (n_a_col != n_b_col) return -1;
		p->n_d = 2, p->d[0] = n_a_row, p->d[1] = n_b_row;
	} else if (action == KAD_FORWARD) {
		//memset(p->x, 0, n_a_row * n_b_row * sizeof(float));
        for (j = 0; j < n_a_row * n_b_row; j++)
            p->x_c[j].clean() = true;
        if (seal_is_encrypted(q[1]))
			kad_sgemm_simple(0, 1, n_a_row, n_b_row, n_col, q[0]->x_c, q[1]->x_c, p->x_c); /* Y = X * trans(W) */
        else
			kad_sgemm_simple(0, 1, n_a_row, n_b_row, n_col, q[0]->x_c, q[1]->x, p->x_c); /* Y = X * trans(W) */
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q[0]) && (q[1]->x))
			kad_sgemm_simple(0, 0, n_a_row, n_col, n_b_row, p->g_c, q[1]->x, q[0]->g_c); /* G_x <- G_y * W */
        else if (kad_is_back(q[0]) && seal_is_encrypted(q[1]))
			kad_sgemm_simple(0, 0, n_a_row, n_col, n_b_row, p->g_c, q[1]->x_c, q[0]->g_c); /* G_x <- G_y * W */
		if (kad_is_back(q[1]) && q[0]->x_c)
			kad_sgemm_simple(1, 0, n_b_row, n_col, n_a_row, p->g_c, q[0]->x_c, q[1]->g_c); /* G_w <- trans(G_y) * X */

	}
	return 0;
}

int kad_op_matmul(kad_node_t *p, int action) /* TODO: matmul and cmul have different broadcasting rules */
{
	int n_a_row, n_b_row, n_a_col, n_b_col;
    int j;
	kad_node_t *q[2];

	q[0] = p->child[0];
	q[1] = p->child[1];
	n_a_row = q[0]->n_d == 1? 1 : q[0]->d[0];
	n_b_row = q[1]->n_d == 1? 1 : q[1]->d[0];
	n_a_col = kad_len(q[0]) / n_a_row;
	n_b_col = kad_len(q[1]) / n_b_row;
	if (action == KAD_SYNC_DIM) {
		if (n_a_col != n_b_row) return -1;
		p->n_d = 2, p->d[0] = n_a_row, p->d[1] = n_b_col;
	} else if (action == KAD_FORWARD) {
		//memset(p->x, 0, n_a_row * n_b_col * sizeof(float));
        for (j = 0; j < n_a_row * n_b_col; j++)
            p->x_c[j].clean() = true;
        if (seal_is_encrypted(q[1]))
			kad_sgemm_simple(0, 0, n_a_row, n_b_col, n_a_col, q[0]->x_c, q[1]->x_c, p->x_c); /* Y = X * W */
        else
			kad_sgemm_simple(0, 0, n_a_row, n_b_col, n_a_col, q[0]->x_c, q[1]->x, p->x_c); /* Y = X * W */
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q[0]) && q[1]->x)
			kad_sgemm_simple(0, 1, n_a_row, n_a_col, n_b_col, p->g_c, q[1]->x, q[0]->g_c); /* G_x <- G_y * trans(W) */
        else if (kad_is_back(q[0]) && seal_is_encrypted(q[1]))
			kad_sgemm_simple(0, 1, n_a_row, n_a_col, n_b_col, p->g_c, q[1]->x_c, q[0]->g_c); /* G_x <- G_y * trans(W) */
        //dont really see the diff between x_c existing and is_encrypted being true.
		if (kad_is_back(q[1]) && q[0]->x_c)
			kad_sgemm_simple(1, 0, n_b_row, n_b_col, n_a_row, q[0]->x_c, p->g_c, q[1]->g_c); /* G_y <- trans(A) * G_y */
	}
	return 0;
}

int kad_op_square(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i)
            seal_multiply(q->x_c[i], q->x_c[i], p->x_c[i]);
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < n; ++i){
            seal_add(q->x_c[i], q->x_c[i], *ciphertext);
            seal_multiply_inplace(*ciphertext, p->g_c[i]);
            seal_add_inplace(q->g_c[i], *ciphertext);
        }
	}
	return 0;
}

int kad_op_1minus(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i){
            engine->get_evaluator()->negate(q->x_c[i].ciphertext(), ciphertext->ciphertext());
            seal_add(*ciphertext, 1.0, p->x_c[i]);
        }
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		kad_saxpy(n, -1.0f, p->g_c, q->g_c);
	}
	return 0;
}

int kad_op_exp(kad_node_t *p, int action)
{
	int i, j, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
        if(remote)
        {
            //not implemented
            assert(false);
        }else{
            for (i = 0; i < n; ++i) {
                engine->decrypt(q->x_c[i], *plaintext);
                engine->decode(*plaintext, t);
                for (j = 0; j < t.size(); j++)
                    t[i] = expf(t[i]);
                engine->encode(t, *plaintext);
                engine->encrypt(*plaintext, p->x_c[i]);
            }
        }
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < n; ++i){
                seal_multiply(p->g_c[i], p->x_c[i], *ciphertext);
                seal_add_inplace(q->g_c[i], *ciphertext);
        }
	}
	return 0;
}

int kad_op_log(kad_node_t *p, int action)
{
	int i, j, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
        if(remote)
        {
            //not implemented
            assert(false);
        }else{
            for (i = 0; i < n; ++i) {
                engine->decrypt(q->x_c[i], *plaintext);
                engine->decode(*plaintext, t);
                for (j = 0; j < t.size(); j++)
                    t[i] = logf(t[i]);
                engine->encode(t, *plaintext);
                engine->encrypt(*plaintext, p->x_c[i]);
            }
        }
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		// input must be encrypted.
		assert(seal_is_encrypted(q));
		if(remote){
			//not implemented
			assert(false);
		}else{
			for (i = 0; i < n; ++i) {
				engine->decrypt(q->x_c[i], *plaintext);
				engine->decode(*plaintext, t);
				for (j = 0; j < t.size(); j++)
					t[i] = 1.0/t[i];
				engine->encode(t, *plaintext);
				engine->encrypt(*plaintext, *ciphertext);
				seal_multiply_inplace(*ciphertext, p->g_c[i]);
				seal_add_inplace(q->g_c[i], *ciphertext);
			}
		}
	}
	return 0;
}


// sum on the (p->ptr) axis.
int kad_op_reduce_sum(kad_node_t *p, int action)
{
	kad_node_t *q = p->child[0];
	int i, j, k, axis, d0, d1;

	assert(p->ptr);
	axis = *(int32_t*)p->ptr;
	if (axis < 0 || axis >= q->n_d) return -1;
	for (i = 0, d0 = 1; i < axis; ++i) d0 *= q->d[i];
	for (i = axis + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
	if (action == KAD_SYNC_DIM) {
		p->n_d = q->n_d - 1;
		for (i = j = 0; i < q->n_d; ++i)
			if (i != axis) p->d[j++] = q->d[i];
	} else if (action == KAD_FORWARD) {
        for (i = 0; i < kad_len(p); ++i)
            p->x_c[i].clean() = true;
		for (i = 0; i < d0; ++i)
			for (j = 0; j < q->d[axis]; ++j)
				for (k = 0; k < d1; ++k)
                    seal_add_inplace(p->x_c[i * d1 + k], q->x_c[(i * q->d[axis] + j) * d1 + k]);
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < d0; ++i)
			for (j = 0; j < q->d[axis]; ++j)
				for (k = 0; k < d1; ++k)
					seal_add_inplace(q->g_c[(i * q->d[axis] + j) * d1 + k], p->g_c[i * d1 + k]);;
	}
	return 0;
}

int kad_op_reduce_mean(kad_node_t *p, int action)
{
	kad_node_t *q = p->child[0];
	int i, j, k, axis, d0, d1;

	assert(p->ptr);
	axis = *(int32_t*)p->ptr;
	if (axis < 0 || axis >= q->n_d) return -1;
	for (i = 0, d0 = 1; i < axis; ++i) d0 *= q->d[i];
	for (i = axis + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
	if (action == KAD_SYNC_DIM) {
		p->n_d = q->n_d - 1;
		for (i = j = 0; i < q->n_d; ++i)
			if (i != axis) p->d[j++] = q->d[i];
	} else if (action == KAD_FORWARD) {
		float coeff = 1.0f / q->d[axis];
        for (i = 0; i < kad_len(p); ++i)
            p->x_c[i].clean() = true;
		for (i = 0; i < d0; ++i)
			for (j = 0; j < q->d[axis]; ++j)
				for (k = 0; k < d1; ++k){
                    seal_add_inplace(p->x_c[i*d1+k], q->x_c[(i*q->d[axis]+j)*d1+k]);
                }
		for (i = 0; i < kad_len(p); ++i)
			seal_multiply_inplace(p->x_c[i], coeff);
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		float t = 1.0f / q->d[axis];
		for (i = 0; i < d0; ++i)
			for (j = 0; j < q->d[axis]; ++j)
				for (k = 0; k < d1; ++k){
                    seal_multiply(p->g_c[i*d1 + k], t, *ciphertext);
                    seal_add_inplace(q->g_c[(i*q->d[axis] + j) * d1 + k], *ciphertext);
                }
	}
	return 0;
}

/********** Miscellaneous **********/

int kad_op_dropout(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	assert(p->child[1]->n_d == 0);
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_ALLOC) {
		if (kad_is_back(p->child[0]))
			p->gtmp = realloc(p->gtmp, n);
	} else if (action == KAD_FORWARD) {
		float r = kad_is_const(q) || kad_is_var(q)? 0.0f : *p->child[1]->x, z = 1.0f / (1.0f - r);
		uint8_t *flag = (uint8_t*)p->gtmp;
		for (i = 0; i < n; ++i) {
			int kept = (kad_drand(p->ptr) >= r);
            if (kept)
                seal_multiply(q->x_c[i], z, p->x_c[i]);
            else
                p->x_c[i].clean() = true;
			if (flag) flag[i] = kept;
		}
	} else if (action == KAD_BACKWARD && kad_is_back(p->child[0])) {
		float r = kad_is_const(q) || kad_is_var(q)? 0.0f : *p->child[1]->x, z = 1.0f / (1.0f - r);
		uint8_t *flag = (uint8_t*)p->gtmp;
		for (i = 0; i < n; ++i)
			if (flag[i]) {
                seal_multiply(p->g_c[i], z, *ciphertext);
                seal_add_inplace(q->g_c[i], *ciphertext);
            }
	}
	return 0;
}


int kad_op_sample_normal(kad_node_t *p, int action) /* not tested */
{assert(false);
	/***
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_ALLOC) {
		if (kad_is_back(p->child[0]))
			p->gtmp = realloc(p->gtmp, n * sizeof(float));
	} else if (action == KAD_FORWARD) {
		float *r = (float*)p->gtmp;
		for (i = 0; i < n; ++i) {
			float z;
			z = (float)kad_drand_normal(p->ptr);
			p->x[i] = q->x[i] * z;
			if (r) r[i] = z;
		}
	} else if (action == KAD_BACKWARD && kad_is_back(p->child[0])) {
		float *r = (float*)p->gtmp;
		for (i = 0; i < n; ++i)
			q->g[i] += p->g[i] * r[i];
	}
	return 0;
	***/
}



int kad_op_slice(kad_node_t *p, int action)
{assert(false);
	/***
	kad_node_t *q = p->child[0];
	int32_t *aux, *range;
	int i, axis, d0, d1;

	assert(p->ptr);
	aux = (int32_t*)p->ptr, axis = aux[0], range = aux + 1;
	if (axis < 0 || axis >= q->n_d) return -1;
	for (i = 0, d0 = 1; i < axis; ++i) d0 *= q->d[i];
	for (i = axis + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
	if (action == KAD_SYNC_DIM) {
		if (range[0] >= range[1] || range[0] < 0 || range[1] > q->d[axis]) return -1;
		kad_copy_dim1(p, q);
		p->d[axis] = range[1] - range[0];
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < d0; ++i)
			std::memcpy(&p->x_c[i * p->d[axis] * d1], &q->x_c[(i * q->d[axis] + range[0]) * d1], (range[1] - range[0]) * d1 * sizeof(SEALCiphertext));
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < d0; ++i)
			kad_saxpy((range[1] - range[0]) * d1, 1.0f, &p->g_c[i * p->d[axis] * d1], &q->g_c[(i * q->d[axis] + range[0]) * d1]);
	}
	***/
	return 0;
}

int kad_op_concat(kad_node_t *p, int action)
{assert(false);
	/***
	kad_node_t *q = p->child[0];
	int32_t *aux;
	int i, j, k, axis, d0, d1;

	assert(p->ptr);
	aux = (int32_t*)p->ptr, axis = aux[0];
	for (i = 0, d0 = 1; i < axis; ++i) d0 *= q->d[i];
	for (i = axis + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
	if (action == KAD_SYNC_DIM) {
		for (i = 1; i < p->n_child; ++i) {
			if (p->child[i]->n_d != q->n_d) return -1;
			for (j = 0; j < q->n_d; ++j)
				if (j != axis && q->d[j] != p->child[i]->d[j]) return -1;
		}
		kad_copy_dim1(p, q);
		for (i = 1; i < p->n_child; ++i)
			p->d[axis] += p->child[i]->d[axis];
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < d0; ++i)
			for (j = k = 0; j < p->n_child; ++j) {
				q = p->child[j];
				std::memcpy(&p->x_c[(i * p->d[axis] + k) * d1], &q->x_c[i * q->d[axis] * d1], q->d[axis] * d1 * sizeof(SEALCiphertext));
				k += q->d[axis];
			}
	} else if (action == KAD_BACKWARD) {
		for (i = 0; i < d0; ++i)
			for (j = k = 0; j < p->n_child; ++j) {
				q = p->child[j];
				if (!kad_is_back(q)) continue;
				kad_saxpy(q->d[axis] * d1, 1.0f, &p->g_c[(i * p->d[axis] + k) * d1], &q->g_c[i * q->d[axis] * d1]);
				k += q->d[axis];
			}
	}
	***/
	return 0;
}

int kad_op_reshape(kad_node_t *p, int action)
{assert(false);
	/***
	kad_node_t *q = p->child[0];

	if (action == KAD_SYNC_DIM) {
		if (p->ptr) {
			int32_t *aux = (int32_t*)p->ptr;
			int i, len = 1, n_missing = 0;
			p->n_d = p->ptr_size / 4;
			for (i = 0; i < p->n_d; ++i) p->d[i] = aux[i];
			for (i = 0; i < p->n_d; ++i)
				if (p->d[i] <= 0) ++n_missing;
				else len *= p->d[i];
			if (n_missing == 0 && len != kad_len(q)) return -1;
			if (n_missing > 1) {  //attempt to infer missing dimensions except the last one 
				for (i = 0; i < p->n_d; ++i)
					if (p->d[i] <= 0 && i < q->n_d) {
						p->d[i] = q->d[i], len *= p->d[i];
						if (--n_missing == 1) break;
					}
				if (n_missing > 1) return -1;
			}
			if (n_missing == 1) { //infer the last missing dimension 
				if (kad_len(q) % len != 0) return -1;
				for (i = 0; i < p->n_d; ++i)
					if (p->d[i] <= 0) p->d[i] = kad_len(q) / len;
			}
		} else kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		std::memcpy(p->x_c, q->x_c, kad_len(p) * sizeof(SEALCiphertext));
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		kad_saxpy(kad_len(p), 1.0f, p->g_c, q->g_c);
	}
	***/
	return 0;
}

int kad_op_reverse(kad_node_t *p, int action)
{
	/***
	kad_node_t *q = p->child[0];
	int axis, i, j, n, d0, d1;

	axis = p->ptr? *(int32_t*)p->ptr : 0;
	if (axis < 0) axis += q->n_d;
	assert(axis >= 0 && axis < q->n_d);
	for (i = 0, d0 = 1; i < axis; ++i) d0 *= q->d[i];
	n = q->d[axis];
	for (i = axis + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < d0; ++i)
			for (j = 0; j < n; ++j)
				std::memcpy(&p->x_c[(i * n + n - 1 - j) * d1], &q->x_c[(i * n + j) * d1], d1 * sizeof(SEALCiphertext));
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < d0; ++i)
			for (j = 0; j < n; ++j)
				kad_saxpy(d1, 1.0f, &p->g_c[(i * n + n - 1 - j) * d1], &q->g_c[(i * n + j) * d1]);
	}
	***/
	return 0;

}

/********** Cost functions **********/


// better transfer all cost functions to bootstrapping functions.

int kad_op_mse(kad_node_t *p, int action)
{
	kad_node_t *y1 = p->child[0]; /* test */
	kad_node_t *y0 = p->child[1]; /* truth */
	int i, n;
	if (action == KAD_SYNC_DIM) {
		p->n_d = 0;
	} else if (action == KAD_FORWARD) {
        if (remote){
            //NOT IMPLEMENTED
            assert(false);
        }else{
            engine->decrypt(y1->x_c[0], *plaintext);
            engine->decode(*plaintext, test_t);
            engine->decrypt(y0->x_c[0], *plaintext);
            engine->decode(*plaintext, truth_t);
            for (i=0; i< truth_t.size(); i++)
                test_t[i] = pow((truth_t[i] - test_t[i]), 2.0) * 1./float(y0->x_c[0].size());
            engine->encode(test_t, *plaintext);
            engine->encrypt(*plaintext, p->x_c[0]);
		}
	} else if (action == KAD_BACKWARD && kad_is_back(y1)) {
        if (remote){
            //NOT IMPLEMENTED
            assert(false);
        }else{
			float coeff = 2.0f * p->g[0]/float(y0->x_c[0].size());
            engine->decrypt(y1->x_c[0], *plaintext);
            engine->decode(*plaintext, test_t);
            engine->decrypt(y0->x_c[0], *plaintext);
            engine->decode(*plaintext, truth_t);
            for (i=0; i < truth_t.size(); i++)
                test_t[i] = coeff * (test_t[i] - truth_t[i]);
            engine->encode(test_t, *plaintext);
            seal_add_inplace(y1->g_c[0], *plaintext);
		}
	}
	return 0;
}

int kad_op_ce_bin(kad_node_t *p, int action)
{
	static const float tiny = 1e-9f;
	kad_node_t *y1 = p->child[0]; /* test */
	kad_node_t *y0 = p->child[1]; /* truth */
	int i, n;
	if (action == KAD_SYNC_DIM) {
		p->n_d = 0;
	} else if (action == KAD_FORWARD) {
        if (remote){
            //NOT IMPLEMENTED
            assert(false);
        }else{
            engine->decrypt(y1->x_c[0], *plaintext);
            engine->decode(*plaintext, test_t);
            engine->decrypt(y0->x_c[0], *plaintext);
            engine->decode(*plaintext, truth_t);
            for (i=0; i<truth_t.size(); i++){
				test_t[i] = (-log(test_t[i]+1e-6)*truth_t[i] + (truth_t[i]-1)*log(1e-6 + 1.-test_t[i]))/(double)(truth_t.size());
			}
            engine->encode(test_t, *plaintext);
            engine->encrypt(*plaintext, p->x_c[0]);
        }
	} else if (action == KAD_BACKWARD && kad_is_back(y1)) {
        if (remote){
            //NOT IMPLEMENTED
            assert(false);
        }else{
            engine->decrypt(y1->x_c[0], *plaintext);
            engine->decode(*plaintext, test_t);
            engine->decrypt(y0->x_c[0], *plaintext);
            engine->decode(*plaintext, truth_t);
            for (i=0; i<truth_t.size(); i++)
                test_t[i] = p->g[0]*(-1./(test_t[i]+1e-6)*truth_t[i] + (1-truth_t[i])*1./(1.-test_t[i]+1e-6))/(float)(truth_t.size());
            engine->encode(test_t, *plaintext);
            seal_add_inplace(y1->g_c[0], *plaintext);
        }
	}
	return 0;
}

int kad_op_ce_bin_neg(kad_node_t *p, int action)//use 1/-1 as labels
{
    //not implemented, because don't know its meaning.
    assert(false);
	return 0;
}

int kad_op_ce_multi(kad_node_t *p, int action)
{
	static const float tiny = 1e-9f;
	kad_node_t *y1 = p->child[0]; /* test */
	kad_node_t *y0 = p->child[1]; /* truth */
	kad_node_t *c = 0;
	int i, j, n1, d0;
	n1 = y0->d[y0->n_d - 1];
	//d0 = kad_len(y0) / n1;
    // this third child is for masking.
	if (p->n_child == 3) {
		c = p->child[2];
		assert(c->n_d == 1 && c->d[0] == n1);
	}
	if (action == KAD_SYNC_DIM) {
		if (kad_len(y0) != kad_len(y1) || y0->d[y0->n_d - 1] != y1->d[y1->n_d - 1]) return -1;
		p->n_d = 0;
	} else if (action == KAD_FORWARD) {
        if (remote){
            assert(false);
        }else{
			int batch_size = y1->x_c[0].size();
            if (c == 0) {
                vector<double> cost(batch_size, 0.);
                for (i = 0; i < n1; ++i){
                    engine->decrypt(y1->x_c[i], *plaintext);
                    engine->decode(*plaintext, test_t);
                    engine->decrypt(y0->x_c[i], *plaintext);
                    engine->decode(*plaintext, truth_t);
                    for (j=0; j<batch_size;j++)
                        cost[j] += -log(test_t[j]+1e-6)*truth_t[j]/(float)(batch_size);
                }
                engine->encode(cost, *plaintext);
                engine->encrypt(*plaintext, p->x_c[0]);
            } else {
                vector<double> cost(batch_size, 0.);
                for (i = 0; i < n1; ++i){
                    engine->decrypt(y1->x_c[i], *plaintext);
                    engine->decode(*plaintext, test_t);
                    engine->decrypt(y0->x_c[i], *plaintext);
                    engine->decode(*plaintext, truth_t);
                    for (j=0; j<batch_size;j++)
                        cost[j] += -c->x[i]*log(test_t[j]+1e-6)*truth_t[j]/(float)(batch_size);
                }
                engine->encode(cost, *plaintext);
                engine->encrypt(*plaintext, p->x_c[0]);
            }
        }
	} else if (action == KAD_BACKWARD && (kad_is_back(y1))) {
        if (remote){
            assert(false);
        }else{
			int batch_size = y1->x_c[0].size();
            float coeff = p->g[0] / (float)(batch_size);
            if (c == 0) {
                for (i = 0; i < n1; ++i){
                    engine->decrypt(y1->x_c[i], *plaintext);
                    engine->decode(*plaintext, test_t);
                    engine->decrypt(y0->x_c[i], *plaintext);
                    engine->decode(*plaintext, truth_t);
                    for (j = 0; j < batch_size; j++)
                        test_t[j] = -coeff * truth_t[j] * 1.0 / (test_t[j]+1e-6);
                    engine->encode(test_t, *plaintext);
                    seal_add_inplace(y1->g_c[i], *plaintext);
                }
            } else {
                for (i = 0; i < n1; ++i){
                    engine->decrypt(y1->x_c[i], *plaintext);
                    engine->decode(*plaintext, test_t);
                    engine->decrypt(y0->x_c[i], *plaintext);
                    engine->decode(*plaintext, truth_t);
                    for (j=0; j<batch_size; j++)
                        test_t[j] = -coeff * c->x[i] * truth_t[j] * 1.0 / (test_t[j]+1e-6);
                    engine->encode(test_t, *plaintext);
                	seal_add_inplace(y1->g_c[i], *plaintext);
                }
		    }
        }
	}
	return 0;
}

/********** Normalization **********/
int kad_op_stdnorm(kad_node_t *p, int action)
{assert(false);
    printf("not implemented");
/***
	int i, j, n, m;
	kad_node_t *q = p->child[0];
	assert(q->n_d > 0);
	n = q->d[q->n_d - 1];
	m = kad_len(q) / n;
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_ALLOC) {
		p->gtmp = realloc(p->gtmp, m * sizeof(float));
	} else if (action == KAD_FORWARD) {
		float *si = (float*)p->gtmp;
		for (j = 0; j < m; ++j) {
			float *px = &p->x[j * n], *qx = &q->x[j * n];
			float avg, std_inv;
			double s;
			for (i = 0, s = 0.0; i < n; ++i) s += qx[i];
			avg = (float)(s / n);
			for (i = 0; i < n; ++i) px[i] = qx[i] - avg;
			for (i = 0, s = 0.0; i < n; ++i) s += px[i] * px[i];
			std_inv = s == 0.0? 1.0f : (float)(1.0 / sqrt(s / n));
			for (i = 0; i < n; ++i) px[i] *= std_inv;
			si[j] = std_inv;
		}
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		float *si = (float*)p->gtmp;
		for (j = 0; j < m; ++j) {
			float *pg = &p->g[j * n], *qg = &q->g[j * n], *px = &p->x[j * n], std_inv = si[j];
			double s, t;
			for (i = 0, s = t = 0.0; i < n; ++i)
				s += pg[i], t += px[i] * pg[i];
			s /= n, t /= n;
			for (i = 0; i < n; ++i)
				qg[i] += std_inv * (pg[i] - s - px[i] * t);
		}
	}
    ***/
	return 0;
}

/********** Activation functions **********/
int kad_op_sigm(kad_node_t *p, int action)
{
	int i, n, j;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
        if (remote){
            assert(false);
        }else{
            for (i = 0; i < n; i++){
                engine->decrypt(q->x_c[i], *plaintext);
                engine->decode(*plaintext, t);
                for (j = 0; j < t.size(); ++j)
                    t[j] = 1.0f / (1.0f + expf(-t[j]));
                engine->encode(t, *plaintext);
                engine->encrypt(*plaintext, p->x_c[i]);
            }
        }
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
        if (remote){
            assert(false);
        }
        else{
			if(p->g_c[i].clean()){
				q->g_c[i].clean() = true;
			}else{			
				for (i=0; i<n; i++){
					engine->decrypt(p->x_c[i], *plaintext);
					engine->decode(*plaintext, test_t);
					engine->decrypt(p->g_c[i], *plaintext);
					engine->decode(*plaintext, t);
					for (j = 0; j < t.size(); ++j)
						test_t[j] = t[j] * (test_t[j] * (1. - test_t[j]));
					engine->encode(test_t, *plaintext);
					seal_add_inplace(q->g_c[i], *plaintext);
				}
			}
        }
	}
	return 0;
}

int kad_op_tanh(kad_node_t *p, int action)
{
	int i, n, j;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
        if (remote){
            assert(false);
        }else{
            for (i = 0; i < n; i++){
                engine->decrypt(q->x_c[i], *plaintext);
                engine->decode(*plaintext, t);
                for (j = 0; j < t.size(); ++j){
                    if (t[j] < -20.0f) t[j] = -1.0f;
                    else {
                        float y;
                        y = expf(-2.0f * t[j]);
                        t[j] = (1.0f - y) / (1.0f + y);
                    }
                }
                engine->encode(t, *plaintext);
                engine->encrypt(*plaintext, p->x_c[i]);
            }
		}
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
        if (remote){
            assert(false);
        }
        else{
			if(p->g_c[i].clean()){
				q->g_c[i].clean() = true;
			}else{			
				for (i=0; i<n; i++){
					engine->decrypt(p->x_c[i], *plaintext);
					engine->decode(*plaintext, test_t);
					engine->decrypt(p->g_c[i], *plaintext);
					engine->decode(*plaintext, t);
					for (j = 0; j < t.size(); ++j)
						test_t[j] = t[j] * (1. - test_t[j] * test_t[j]);
					engine->encode(test_t, *plaintext);
					seal_add_inplace(q->g_c[i], *plaintext);
				}
			}
        }
	}
	return 0;
}

int kad_op_relu(kad_node_t *p, int action)
{
	int i, n, j;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
        if (remote){
            assert(false);
        }else{
            for (i = 0; i < n; i++){
                engine->decrypt(q->x_c[i], *plaintext);
                engine->decode(*plaintext, t);
                for (j = 0; j < t.size(); ++j)
                    t[j] = t[j] > 0.0f? t[j] : 0.0f;
                engine->encode(t, *plaintext);
                engine->encrypt(*plaintext, p->x_c[i]);
            }
        }
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
        if (remote){
            assert(false);
        }
        else{
            for (i=0; i<n; i++){
				if(p->g_c[i].clean()){
					q->g_c[i].clean() = true;
				}else{
					engine->decrypt(p->x_c[i], *plaintext);
					engine->decode(*plaintext, test_t);
					engine->decrypt(p->g_c[i], *plaintext);
					engine->decode(*plaintext, t);
					for (j = 0; j < t.size(); ++j)
						test_t[j] = test_t[j] > 0.0f? t[j] : 0.0f;
					engine->encode(test_t, *plaintext);
					engine->encrypt(*plaintext, *ciphertext);
					seal_add_inplace(q->g_c[i], *ciphertext);					
				}

            }
        }
	}
	return 0;
}

int kad_op_sin(kad_node_t *p, int action)
{
	int i, n, j;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
        if (remote){
            assert(false);
        }else{
            for (i = 0; i < n; i++){
                engine->decrypt(q->x_c[i], *plaintext);
                engine->decode(*plaintext, t);
                for (j = 0; j < t.size(); ++j)
                    t[j] = sinf(t[j]);
                engine->encode(t, *plaintext);
                engine->encrypt(*plaintext, p->x_c[i]);
            }
        }
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
        if (remote){
            assert(false);
        }
        else{
			if(p->g_c[i].clean()){
				q->g_c[i].clean() = true;
			}else{
				for (i=0; i<n; i++){
					engine->decrypt(q->x_c[i], *plaintext);
					engine->decode(*plaintext, test_t);
					engine->decrypt(p->g_c[i], *plaintext);
					engine->decode(*plaintext, t);
					for (j = 0; j < t.size(); ++j)
						test_t[j] = t[j]*cosf(test_t[j]);
					engine->encode(test_t, *plaintext);
					seal_add_inplace(q->g_c[i], *plaintext);
				}
			}
        }
	}
	return 0;
}

int kad_op_softmax(kad_node_t *p, int action)
{
	int i, j, n1, d0;
	kad_node_t *q = p->child[0];
	float s;
	int batch_size = q->x_c[0].size();

	n1 = q->d[q->n_d - 1];
	//d0 = kad_len(q) / n1;
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
        if (remote){
            assert(false);
        }else{
            vector<vector<double>> raw_value(n1, vector<double>(batch_size, 0.));
            vector<double> &y = test_t;
            vector<double> &max = truth_t; 
			y.resize(batch_size);
			max.resize(batch_size);
            for (j = 0; j < batch_size; ++j)
                max[j] = -FLT_MAX;
            for (i = 0; i < n1; i++){
                engine->decrypt(q->x_c[i],*plaintext);
                engine->decode(*plaintext,raw_value[i]);
                for (j = 0; j < batch_size; ++j)
                    max[j] = max[j] > raw_value[i][j]? max[j] : raw_value[i][j];
            }
            for (j = 0; j < batch_size; ++j){
                for (i = 0, s = 0.0f; i < n1; i++){
                    raw_value[i][j] = expf(raw_value[i][j]-max[j]);
                    s += raw_value[i][j];
                }
                for (i = 0, s = 1.0f /s; i < n1; ++i)
                    raw_value[i][j] *= s;
            }
            for (i = 0; i < n1; i++){
                engine->encode(raw_value[i], *plaintext);
                engine->encrypt(*plaintext, p->x_c[i]);
            }
        }
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
        if (remote){
            assert(false);
        }else{
            vector<vector<double>> raw_value(n1, vector<double>(batch_size, 0.));
            vector<vector<double>> raw_gradient(n1, vector<double>(batch_size, 0.));
            for (i = 0; i < n1; i++){
                engine->decrypt(p->x_c[i],*plaintext);
                engine->decode(*plaintext,raw_value[i]);
                engine->decrypt(p->g_c[i],*plaintext);
                engine->decode(*plaintext,raw_gradient[i]);
            }
			// for xi, the gradient is g1*y1*(1-y1)-g2*y1*y2-g3*y1*y3 = yi(gi-(\Sigma_i y_i*g_i))
            for (j = 0; j < batch_size; ++j){
                for (i = 0, s = 0.0f; i < n1; i++)
                    s += raw_gradient[i][j] * raw_value[i][j];
                for (i = 0; i < n1; i ++)
                    raw_value[i][j] = raw_value[i][j]*(raw_gradient[i][j] - s);
            }
            for (i = 0; i < n1; i++){
                engine->encode(raw_value[i], *plaintext);
                seal_add_inplace(q->g_c[i], *plaintext);
            } 
        }
	}
	return 0;
}

/********** Multi-node pooling **********/

// we need this.
int kad_op_avg(kad_node_t *p, int action)
{
	assert(false);
    //not implemented
    //
    /***
	int i, n;
	float tmp;
	kad_node_t *q;

	assert(p->n_child > 0);
	tmp = 1.0f / p->n_child;
	q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		for (i = 1; i < p->n_child; ++i)
			if (kad_len(p->child[i]) != n) return -1;
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		std::memcpy(p->x, q->x, n * sizeof(float));
		for (i = 1; i < p->n_child; ++i)
			kad_saxpy(n, 1.0f, p->child[i]->x, p->x);
		for (i = 0; i < n; ++i) p->x[i] *= tmp;
	} else if (action == KAD_BACKWARD) {
		for (i = 0; i < p->n_child; ++i)
			if (kad_is_back(p->child[i]))
				kad_saxpy(n, tmp, p->g, p->child[i]->g);
	}
    ***/
	return 0;
}

int kad_op_max(kad_node_t *p, int action)
{
	assert(false);
    //not implemented
    /***
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		int *max_j;
		for (i = 1; i < p->n_child; ++i)
			if (kad_len(p->child[i]) != n) return -1;
		kad_copy_dim1(p, q);
		max_j = (int*)calloc(n, sizeof(int));
		p->gtmp = max_j;
	} else if (action == KAD_FORWARD) {
		int j, *max_j = (int*)p->gtmp;
		memset(max_j, 0, n * sizeof(int));
		std::memcpy(p->x, q->x, n * sizeof(float));
		for (j = 1; j < p->n_child; ++j)
			for (i = 0, q = p->child[j]; i < n; ++i)
				if (q->x[i] > p->x[i]) p->x[i] = q->x[i], max_j[i] = j;
	} else if (action == KAD_BACKWARD) {
		int *max_j = (int*)p->gtmp;
		for (i = 0; i < n; ++i)
			p->child[max_j[i]]->g[i] += p->g[i];
	}
    ***/
	return 0;
}

int kad_op_stack(kad_node_t *p, int action) /* TODO: allow axis, as in TensorFlow */
{
    //not implemented for now
	assert(false);
	/**
	int i, n, axis = 0;
	kad_node_t *q;

	assert(p->n_child > 0);
	q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		for (i = 1; i < p->n_child; ++i)
			if (kad_len(p->child[i]) != n) return -1;
		p->n_d = q->n_d + 1;
		for (i = 0; i < axis; ++i) p->d[i] = q->d[i];
		p->d[axis] = p->n_child;
		for (; i < q->n_d; ++i) p->d[i+1] = q->d[i];
	} else if (action == KAD_FORWARD) { // TODO: doesn't work when axis != 0 
		for (i = 0; i < p->n_child; ++i)
			std::memcpy(&p->x[i * n], p->child[i]->x, n * sizeof(float));
	} else if (action == KAD_BACKWARD) {
		for (i = 0; i < p->n_child; ++i)
			if (kad_is_back(p->child[i]))
				kad_saxpy(n, 1.0f, &p->g[i * n], p->child[i]->g);
	}
	**/
	return 0;
}

int kad_op_select(kad_node_t *p, int action)
{
	kad_node_t *q;
	int i, n, which;

	which = *(int32_t*)p->ptr;
	if (which < 0) which += p->n_child;
	assert(which >= 0 && which < p->n_child);
	q = p->child[which];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		for (i = 0; i < p->n_child; ++i)
			if (p->child[i]->n_d != q->n_d || kad_len(p->child[i]) != n)
				break;
		if (i < p->n_child) return -1;
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		std::memcpy(p->x_c, q->x_c, n * sizeof(SEALCiphertext));
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		kad_saxpy(n, 1.0f, p->g_c, q->g_c);
	}
	return 0;
}

/********** 2D convolution **********/

static void conv_rot180(int d0, int d1, float *x) /* rotate/reverse a weight martix */
{
	int i, j;
	for (i = 0; i < d0; ++i) {
		float tmp, *xi = &x[i * d1];
		for (j = 0; j < d1>>1; ++j)
			tmp = xi[j], xi[j] = xi[d1-1-j], xi[d1-1-j] = tmp; 
	}
}

static void conv_rot180(int d0, int d1, SEALCiphertext *x) /* rotate/reverse a weight martix */
{
	int i, j;
	for (i = 0; i < d0; ++i) {
		SEALCiphertext tmp(engine), *xi = &x[i * d1];
		for (j = 0; j < d1>>1; ++j)
			tmp = xi[j], xi[j] = xi[d1-1-j], xi[d1-1-j] = tmp; 
	}
}


#define conv_out_size(in_size, aux) (((in_size) - (aux)->kernel_size + (aux)->pad[0] + (aux)->pad[1]) / (aux)->stride + 1)

#define process_row_for(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
	int j, l; \
	if (_stride > 1) { \
		for (l = 0; l < _wn; ++l) { \
			const SEALCiphertext *xl = &_xx[l - _pad]; \
			for (j = 0; j < _pn; ++j, xl += _stride) _t[j] = *xl; \
			kad_saxpy(_pn, _ww[l], _t, _yy); \
		} \
	} else for (l = 0; l < _wn; ++l) { \
		kad_saxpy(_pn, _ww[l], &_xx[l - _pad], _yy); \
	} \
} while (0)

#define process_row_back_x(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
	int j, l, k; \
	if (_stride > 1) { \
		for (l = 0; l < _wn; ++l) { \
			SEALCiphertext *xl = &_xx[l - _pad]; \
            for (k = 0; k < _pn; k++) \
                _t[k].clean() = true; \
			kad_saxpy(_pn, _ww[l], _yy, _t); \
			for (j = 0; j < _pn; ++j, xl += _stride) seal_add_inplace(*xl, _t[j]); \
		} \
	} else for (l = 0; l < _wn; ++l) kad_saxpy(_pn, _ww[l], _yy, &_xx[l - _pad]); \
} while (0)

#define process_row_back_w(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
	int j, l; \
	if (_stride > 1) { \
		for (l = 0; l < _wn; ++l) { \
			const SEALCiphertext *xl = &_xx[l - _pad]; \
			for (j = 0; j < _pn; ++j, xl += _stride) _t[j] = *xl; \
			*ciphertext = kad_sdot(_pn, _yy, _t); \
            seal_add_inplace(_ww[l], *ciphertext); \
		} \
	} else for (l = 0; l < _wn; ++l){ \
        *ciphertext = kad_sdot(_pn, _yy, &_xx[l - _pad]); \
        seal_add_inplace(_ww[l], *ciphertext); \
    } \
} while (0)

/* Forward and backward passes are implemented with two different algorithms.
 * The first is faster for small kernels with few input channels; otherwise the
 * second algorithm is faster. Both algorithms should produce identical
 * results, up to the precision of "float".
 */

/* henry 2020.10.6 just implement the CWH method: the first method. */


int kad_op_conv2d(kad_node_t *p, int action) /* in the number-channel-height-width (CHW) shape */
{
#define conv2d_loop1(_x, _w, _y, _tmp, _row_func) do { /* for the CHW shape */ \
		int n, c1, c0, i, k, ii, j; \
        for (c1 = 0; c1 < w->d[0]; ++c1) /* output channel */ \
            for (c0 = 0; c0 < w->d[1]; ++c0) /* input channel */ \
                for (k = 0; k < w->d[2]; ++k) { /* kernel row */ \
                    auto _ww = &(_w)[((c1 * w->d[1] + c0) * w->d[2] + k) * w->d[3]]; \
                    for (i = 0, ii = k - aux[0].pad[0]; i < p->d[1] && ii >= 0 && ii < q->d[1]; ++i, ii += aux[0].stride) { /* output row */ \
                        SEALCiphertext * _xx = &(_x)[((n * q->d[0] + c0) * q->d[1] + ii) * q->d[2]]; \
                        SEALCiphertext * _yy = &(_y)[((n * p->d[0] + c1) * p->d[1] + i)  * p->d[2]]; \
                        if (x_padded) { \
							for (j = 0; j < q->d[2]; j ++) \
								x_padded[aux[1].pad[0] + j] = _xx[j]; \
                            _xx = x_padded + aux[1].pad[0]; \
                        } \
						_row_func(_xx, _ww, _yy, w->d[3], p->d[2], aux[1].stride, aux[1].pad[0], (_tmp)); \
                    } /* ~i */ \
                } /* ~k, c0, c1, n */ \
	} while (0)

    int l ,i ;
	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0], *w = p->child[1];
	SEALCiphertext *t1 = 0, *q1 = 0, *w1 = 0, *x_padded = 0;

	if (action == KAD_FORWARD || action == KAD_BACKWARD) { /* allocate working space */
        t1 = new SEALCiphertext[p->d[2]];
        x_padded = aux[1].pad[0] + aux[1].pad[1] > 0? new SEALCiphertext[q->d[2] + aux[1].pad[0] + aux[1].pad[1]] : 0;
		if(x_padded){
			for (i = 0 ; i < q->d[2] + aux[1].pad[0] + aux[1].pad[1]; i ++){
				x_padded[i].init(engine);
				x_padded[i].clean() = true;
			}
		}
	}
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 3 || w->n_d != 4) return -1;
		if (q->d[0] != w->d[1]) return -1; /* unmatched input channels */
		p->n_d = 3;
		p->d[0] = w->d[0], p->d[1] = conv_out_size(q->d[1], &aux[0]), p->d[2] = conv_out_size(q->d[2], &aux[1]);
	} else if (action == KAD_FORWARD) {
        if(seal_is_encrypted(w)){
		    conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x_c);
			for (l = 0; l < kad_len(q); l++)
                if(!q->x_c[l].clean()){
					engine->decrypt(q->x_c[l], *plaintext);
					engine->decode(*plaintext, truth_t);
				}
			for (l = 0; l < kad_len(w); l++)
                if(!w->x_c[l].clean()){
					engine->decrypt(w->x_c[l], *plaintext);
					engine->decode(*plaintext, truth_t);
				}
            for (l = 0; l < kad_len(p); l++)
                p->x_c[l].clean() = true;
            conv2d_loop1(q->x_c, w->x_c, p->x_c, t1, process_row_for);
            conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x_c);
			
        }else{
		    conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
            for (l = 0; l < kad_len(p); l++)
                p->x_c[l].clean() = true;
            conv2d_loop1(q->x_c, w->x, p->x_c, t1, process_row_for);
            conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
		}
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(p->child[0])) { /* backprop to the input array */
            if(seal_is_encrypted(w)){
                conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x_c);
                conv2d_loop1(q->g_c, w->x_c, p->g_c, t1, process_row_back_x);
                conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x_c);

            }else{
                conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
                conv2d_loop1(q->g_c, w->x, p->g_c, t1, process_row_back_x);
                conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);

            }
		}
		if (kad_is_back(p->child[1])) { /* backprop to the weight matrix */
            conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->g_c);
            conv2d_loop1(q->x_c, w->g_c, p->g_c, t1, process_row_back_w);
            conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->g_c);
		}
	}
	std::free(q1); std::free(w1); 
	delete[] x_padded;
	delete[] t1;
	return 0;
}

int kad_op_max2d(kad_node_t *p, int action)
{
	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0];
	int batch_size = q->x_c[0].size();
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 3) return -1;
		p->n_d = 3;
		p->d[0] = q->d[0], p->d[1] = conv_out_size(q->d[1], &aux[0]), p->d[2] = conv_out_size(q->d[2], &aux[1]);
	} else if (action == KAD_ALLOC) {
		p->gtmp = realloc(p->gtmp, kad_len(p) * sizeof(int));
	} else if (action == KAD_FORWARD) {
        if (remote){
            assert(false);
        }else{
            int rest = 1, len, t, i;
            int *f = (int*)p->gtmp;
            len = kad_len(p);
            for (i = 0; i < len; ++i) p->x_c[i].clean() = true;
            for (i = 0; i < p->n_d - 2; ++i) rest *= p->d[i];
            for (t = 0; t < rest; ++t) {
                int i, j, k, l, m, p_row = p->d[p->n_d - 2], p_col = p->d[p->n_d - 1];

                for (i = 0; i < p_row; ++i)
                    for (j = 0; j < p_col; ++j){
                        vector<double> max(batch_size, -FLT_MAX);
                        int out_po = (t * p_row + i) * p_col + j;
                        for (k = 0; k < aux[0].kernel_size; ++k){
                            for (l = 0; l < aux[1].kernel_size; ++l){
								//ii: current row 
                                int ii = i * aux[0].stride + k - aux[0].pad[0];
                                if (ii < 0 || ii >= q->d[p->n_d-2]) continue;
								//v0: starting index in the current row
								//v_end: ending index in the current row
								//in_po: current input index
								//out_po: current output index
								//max: current max batch
								//f: current used input index
                                int v0 = (t * q->d[p->n_d - 2] + ii) * q->d[p->n_d - 1];
                                int v_end = v0 + q->d[p->n_d - 1];
                                int in_po = v0 + (l > aux[1].pad[0]? l - aux[1].pad[0] : 0) + aux[1].stride * j;
                                if (in_po >= v_end) continue;
                                engine->decrypt(q->x_c[in_po], *plaintext);
                                engine->decode(*plaintext, test_t);
                                for ( m = 0; m < batch_size; m++)
                                    if ( test_t[m] > max[m])
                                        max[m] = test_t[m], f[out_po] = in_po;
                            }
						}
                        engine->encode(max, *plaintext);
                        engine->encrypt(*plaintext, p->x_c[out_po]); 
                    }
            }
        }
	} else if (action == KAD_BACKWARD) {
		int i, len, *f = (int*)p->gtmp;
		len = kad_len(p);
		for (i = 0; i < len; ++i) seal_add_inplace(q->g_c[f[i]],p->g_c[i]);
	}
	return 0;
}


int kad_op_conv1d(kad_node_t *p, int action) /* in the number-channel-width (NCW) shape */
{assert(false);
	/**
#define conv1d_loop1(_x, _w, _y, _tmp, _row_func) do { // for the NCW shape \
		int n, c1, c0; \
		for (n = 0; n < q->d[0]; ++n) //* mini-batch  \
			for (c1 = 0; c1 < w->d[0]; ++c1) //* output channel  \
				for (c0 = 0; c0 < w->d[1]; ++c0) { //* input channel  \
					float *_ww = &(_w)[(c1 * w->d[1] + c0) * w->d[2]]; \
					float *_xx = &(_x)[(n  * q->d[1] + c0) * q->d[2]]; \
					float *_yy = &(_y)[(n  * p->d[1] + c1) * p->d[2]]; \
					if (x_padded) { \
						memcpy(x_padded + aux->pad[0], _xx, q->d[2] * sizeof(float)); \
						_xx = x_padded + aux->pad[0]; \
					} \
					_row_func(_xx, _ww, _yy, w->d[2], p->d[2], aux->stride, aux->pad[0], (_tmp)); \
				} //* ~c0, c1, n  \
	} while (0)

#define conv1d_loop2(_x, _w, _y, _code) do { //* for the NWC shape  \
		int n, c1, j, j_skip = aux->stride * q->d[1], m = w->d[2] * w->d[1]; \
		for (n = 0; n < q->d[0]; ++n) //* mini-batch  \
			for (c1 = 0; c1 < w->d[0]; ++c1) { //* output channel  \
				float *_ww = &(_w)[c1 * m]; \
				float *_xx = &(_x)[n * q->d[1] * q->d[2]]; \
				float *_yy = &(_y)[(n * p->d[1] + c1) * p->d[2]]; \
				if (x_padded) { \
					memcpy(x_padded + aux->pad[0] * q->d[1], _xx, q->d[2] * q->d[1] * sizeof(float)); \
					_xx = x_padded; \
				} \
				for (j = 0; j < p->d[2]; ++j, _xx += j_skip, ++_yy) _code; \
			} //* ~c1, n  \
	} while (0)

	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0], *w = p->child[1];
	float *t = 0, *q1 = 0, *w1 = 0, *x_padded = 0;
	int algo_switch = 0;

	if (action == KAD_FORWARD || action == KAD_BACKWARD) { //* allocate working space 
		if (w->d[2] * w->d[1] < 32) {
			t = (float*)malloc(p->d[2] * sizeof(float));
			x_padded = aux->pad[0] + aux->pad[1] > 0? (float*)calloc(q->d[2] + aux->pad[0] + aux->pad[1], sizeof(float)) : 0;
		} else {
			q1 = (float*)malloc(kad_len(q) * sizeof(float));
			w1 = (float*)malloc(kad_len(w) * sizeof(float));
			x_padded = aux->pad[0] + aux->pad[1] > 0? (float*)calloc((q->d[2] + aux->pad[0] + aux->pad[1]) * q->d[1], sizeof(float)) : 0;
			algo_switch = 1;
		}
	}
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 3 || w->n_d != 3) return -1;
		if (q->d[1] != w->d[1]) return -1; //* unmatched input channels 
		p->n_d = 3;
		p->d[0] = q->d[0], p->d[1] = w->d[0], p->d[2] = conv_out_size(q->d[2], aux);
	} else if (action == KAD_FORWARD) {
		conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
		memset(p->x, 0, kad_len(p) * sizeof(float));
		if (!algo_switch) { //* this is the first algorithm 
			conv1d_loop1(q->x, w->x, p->x, t, process_row_for);
		} else { //* this is the second algorithm 
			conv1d_move_1to2(q->d, q->x, q1);
			conv1d_move_1to2(w->d, w->x, w1);
			conv1d_loop2(q1, w1, p->x, (*_yy += kad_sdot(m, _ww, _xx)));
		}
		conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(p->child[0])) { //* backprop to the input array 
			conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
			if (!algo_switch) {
				conv1d_loop1(q->g, w->x, p->g, t, process_row_back_x);
			} else {
				memset(q1, 0, kad_len(q) * sizeof(float));
				conv1d_move_1to2(w->d, w->x, w1);
				conv1d_loop2(q1, w1, p->g, kad_saxpy(m, *_yy, _ww, _xx));
				conv1d_add_2to1(q->d, q1, q->g);
			}
			conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
		}
		if (kad_is_back(p->child[1])) { //* backprop to the weight matrix 
			conv_rot180(w->d[0] * w->d[1], w->d[2], w->g);
			if (!algo_switch) {
				conv1d_loop1(q->x, w->g, p->g, t, process_row_back_w);
			} else {
				conv1d_move_1to2(q->d, q->x, q1);
				memset(w1, 0, kad_len(w) * sizeof(float));
				conv1d_loop2(q1, w1, p->g, kad_saxpy(m, *_yy, _xx, _ww));
				conv1d_add_2to1(w->d, w1, w->g);
			}
			conv_rot180(w->d[0] * w->d[1], w->d[2], w->g);
		}
	}
	free(t); free(q1); free(w1); free(x_padded);
	***/
	return 0;
}


int kad_op_max1d(kad_node_t *p, int action)
{assert(false);
	/***
	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0];
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 3) return -1;
		p->n_d = 3;
		p->d[0] = q->d[0], p->d[1] = q->d[1], p->d[2] = conv_out_size(q->d[2], aux);
	} else if (action == KAD_ALLOC) {
		p->gtmp = realloc(p->gtmp, kad_len(p) * sizeof(int));
	} else if (action == KAD_FORWARD) {
		int rest = 1, len, t, i;
		int *f = (int*)p->gtmp;
		len = kad_len(p);
		for (i = 0; i < len; ++i) p->x[i] = -FLT_MAX;
		for (i = 0; i < p->n_d - 1; ++i) rest *= p->d[i];
		for (t = 0; t < rest; ++t) {
			int j, l, p_width = p->d[p->n_d - 1];
			int u = t * p_width, v, v0 = t * q->d[p->n_d - 1], v_end = v0 + q->d[p->n_d - 1];
			for (l = 0; l < aux->kernel_size; ++l)
				for (j = 0, v = v0 + (l > aux->pad[0]? l - aux->pad[0] : 0); j < p_width && v < v_end; ++j, v += aux->stride)
					if (p->x[u + j] < q->x[v])
						p->x[u + j] = q->x[v], f[u + j] = v;
		}
	} else if (action == KAD_BACKWARD) {
		int i, len, *f = (int*)p->gtmp;
		len = kad_len(p);
		for (i = 0; i < len; ++i) q->g[f[i]] += p->g[i];
	}***/
	return 0;
}

int kad_op_avg1d(kad_node_t *p, int action)
{assert(false);
	/***
	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0];
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 3) return -1;
		p->n_d = 3;
		p->d[0] = q->d[0], p->d[1] = q->d[1], p->d[2] = conv_out_size(q->d[2], aux);
	} else if (action == KAD_ALLOC) {
		p->gtmp = realloc(p->gtmp, kad_len(p) * sizeof(int));
	} else if (action == KAD_FORWARD) {
		int rest = 1, len, t, i;
		int *f = (int*)p->gtmp;
		len = kad_len(p);
		for (i = 0; i < len; ++i) p->x[i] = 0.0f, f[i] = 0;
		for (i = 0; i < p->n_d - 1; ++i) rest *= p->d[i];
		for (t = 0; t < rest; ++t) {
			int j, l, p_width = p->d[p->n_d - 1];
			int u = t * p_width, v, v0 = t * q->d[p->n_d - 1], v_end = v0 + q->d[p->n_d - 1];
			for (l = 0; l < aux->kernel_size; ++l)
				for (j = 0, v = v0 + (l > aux->pad[0]? l - aux->pad[0] : 0); j < p_width && v < v_end; ++j, v += aux->stride)
					p->x[u + j] += q->x[v], ++f[u + j];
		}
		for (i = 0; i < len; ++i) p->x[i] /= f[i];
	} else if (action == KAD_BACKWARD) {
		int rest = 1, t, i;
		int *f = (int*)p->gtmp;
		for (i = 0; i < p->n_d - 1; ++i) rest *= p->d[i];
		for (t = 0; t < rest; ++t) {
			int j, l, p_width = p->d[p->n_d - 1];
			int u = t * p_width, v, v0 = t * q->d[p->n_d - 1], v_end = v0 + q->d[p->n_d - 1];
			for (l = 0; l < aux->kernel_size; ++l)
				for (j = 0, v = v0 + (l > aux->pad[0]? l - aux->pad[0] : 0); j < p_width && v < v_end; ++j, v += aux->stride)
					q->g[v] += p->g[u + j] / f[u + j];
		}
	}***/
	return 0;
}

/********** List of operators **********/

kad_op_f kad_op_list[KAD_MAX_OP] = {
	0,
	kad_op_add,        /* 1:  element-wise addition */
	kad_op_mul,        /* 2:  element-wise multiplication */
	kad_op_cmul,       /* 3:  column multiplication */
	kad_op_ce_bin_neg, /* 4:  binary cross-entropy for (-1,1) */
	kad_op_square,     /* 5:  square */
	kad_op_sigm,       /* 6:  sigmoid */
	kad_op_tanh,       /* 7:  tanh */
	kad_op_relu,       /* 8:  ReLU */
	kad_op_matmul,     /* 9:  matrix multiplication */
	kad_op_avg,        /* 10: general average pooling (not for ConvNet) */
	kad_op_1minus,     /* 11: 1-x */
	kad_op_select,     /* 12: choose between one of the children */
	kad_op_ce_multi,   /* 13: multi-class cross-entropy */
	kad_op_softmax,    /* 14: softmax */
	kad_op_dropout,    /* 15: dropout */
	kad_op_conv2d,     /* 16: 2D convolution */
	kad_op_max2d,      /* 17: 2D max pooling (for 2D ConvNet) */
	kad_op_conv1d,     /* 18: 1D convolution */
	kad_op_max1d,      /* 19: 1D max pooling (for 1D ConvNet) */
	kad_op_slice,      /* 20: slice data at a dimension */
	kad_op_max,        /* 21: general max pooling */
	kad_op_ce_bin,     /* 22: binary cross-entropy for (0,1) */
	kad_op_sub,        /* 23: element-wise subtraction */
	kad_op_sample_normal,  /* 24: sample from a normal distribution */
	kad_op_reduce_sum,     /* 25 */
	kad_op_reduce_mean,    /* 26 */
	kad_op_log,        /* 27: log() */
	kad_op_avg1d,      /* 28: 1D average pooling (for 1D ConvNet) */
	kad_op_mse,        /* 29: mean square error */
	kad_op_reshape,    /* 30 */
	kad_op_concat,     /* 31 */
	kad_op_stdnorm,    /* 32: layer normalization */
	kad_op_exp,        /* 33: exp() */
	kad_op_sin,        /* 34: sin() */
	kad_op_stack,      /* 35: tf.stack, but on the first axis only */
	kad_op_reverse     /* 36: tf.reverse, but on one axis only */
};



char *kad_op_name[KAD_MAX_OP] = {
	0, "add", "mul", "cmul", "ce_bin_neg", "square", "sigm", "tanh", "relu", "matmul", "avg", "1minus", "select", "ce_multi", "softmax",
	"dropout", "conv2d", "max2d", "conv1d", "max1d", "slice", "max", "ce_bin", "sub", "sample_normal", "reduce_sum", "reduce_mean", "log",
	"avg1d", "mse", "reshape", "concat", "stdnorm", "exp", "sin", "stack", "reverse"
};

/**************************
 *** Debugging routines ***
 **************************/
/**
void kad_trap_fe(void)
{
#ifdef __SSE__
	_MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~(_MM_MASK_INVALID | _MM_MASK_DIV_ZERO));
#endif
}
**/
void kad_print_graph(FILE *fp, int n, kad_node_t **v)
{
	int i, j;
	for (i = 0; i < n; ++i) v[i]->tmp = i;
	for (i = 0; i < n; ++i) {
		kad_node_t *p = v[i];
		std::fprintf(fp, "%d\t%x:%x\t%d\t", i, p->flag, p->ext_flag, p->ext_label);
		if (p->pre) std::fprintf(fp, "%d\t", p->pre->tmp);
		else std::fprintf(fp, ".\t");
		fputs("[", fp);
		for (j = 0; j < p->n_d; ++j) {
			if (j) fputc(',', fp);
			std::fprintf(fp, "%d", p->d[j]);
		}
		std::fprintf(fp, "]\t");
		if (p->n_child) {
			std::fprintf(fp, "%s(", kad_op_name[p->op]);
			for (j = 0; j < p->n_child; ++j) {
				if (j) fputc(',', fp);
				std::fprintf(fp, "$%d", p->child[j]->tmp);
			}
			std::fprintf(fp, ")");
		} else {
            std::fprintf(fp, "%s", kad_is_feed(p)? "feed" : kad_is_var(p)? "var" : kad_is_const(p)? "const" : "N/A");
            std::fprintf(fp, "%s", seal_is_encrypted(p)? "encrypted" : "plain");
		}
		fputc('\n', fp);
	}
	for (i = 0; i < n; ++i) v[i]->tmp = 0;
}

static void kad_add_delta(int n, kad_node_t **a, float c, float *delta)
{
	int i, k;
	for (i = k = 0; i < n; ++i)
		if (kad_is_var(a[i])) {
            if (seal_is_encrypted(a[i])){
                kad_saxpy(kad_len(a[i]), c, &delta[k], a[i]->x_c);
            }else{
                kad_saxpy(kad_len(a[i]), c, &delta[k], a[i]->x);
            }
			k += kad_len(a[i]);
		}
}

void kad_check_grad(int n, kad_node_t **a, int from)
{
	const float eps = 1e-5f, rel = 1e-7f / eps;
	int i, k, j, n_var;
	float *g0, *delta, s0, s1, rel_err, p_m_err,f_minus, f_plus, f0;
	SEALCiphertext f_minus_c(engine), f_plus_c(engine), f0_c(engine);
    SEALCiphertext *g0_c;
	n_var = kad_size_var(n, a);
	g0 = (float*)calloc(n_var, sizeof(float));
	f0_c = *kad_eval_at(n, a, from);
	kad_grad(n, a, from, false);   //no need for you to fp/bp.
	// collect plain gradients into g0
	for (i = k = 0; i < n; ++i){
		if (kad_is_var(a[i])) {
            if (seal_is_encrypted(a[i])){
                for (j = 0; j < kad_len(a[i]); j++){
                    // should be size 1
					engine->decrypt(a[i]->g_c[j], *plaintext);
					engine->decode(*plaintext, t);
					g0[k+j] = t[0];
				}

            } else {
				for (j = 0; j < kad_len(a[i]); j++)
					memcpy(&g0[k], a[i]->g, kad_len(a[i]) * sizeof(float));
			}
			k += kad_len(a[i]);
		}
	}
	delta = (float*)calloc(n_var, sizeof(float));
	for (k = 0; k < n_var; ++k) delta[k] = (float)kad_drand(0) * eps;
	kad_add_delta(n, a, 1.0f, delta);
	f_plus_c = *kad_eval_at(n, a, from);
	kad_add_delta(n, a, -2.0f, delta);
	f_minus_c = *kad_eval_at(n, a, from);
	kad_add_delta(n, a, 1.0f, delta);
	s0 = kad_sdot(n_var, g0, delta);
	//get decrypted s1, firstly we need to sum the batch cost function. (model doesn't do that for efficientcy consideration.)
	sum_vector(f_plus_c);
	engine->decrypt(f_plus_c, *plaintext);
	engine->decode(*plaintext, t);
	f_plus = t[0];
	sum_vector(f_minus_c);
	engine->decrypt(f_minus_c, *plaintext);
	engine->decode(*plaintext, t);
	f_minus = t[0];
	sum_vector(f0_c);
	engine->decrypt(f0_c, *plaintext);
	engine->decode(*plaintext, t);
	f0 = t[0];

	s1 = .5f * (f_plus - f_minus);
	std::fprintf(stderr, "Gradient check -- %g <=> %g @ %g -- ", s0/eps, s1/eps, f0);
	if (fabs(s1) >= rel * eps) {
		rel_err = fabsf(fabsf(s0) - fabsf(s1)) / (fabsf(s0) + fabsf(s1));
		p_m_err = fabsf(f_plus + f_minus - 2.0f * f0) / fabsf(f_plus - f_minus);
		std::fprintf(stderr, "rel_err:%g p_m_err:%g -- ", rel_err, p_m_err);
		if (rel_err >= rel && rel_err > p_m_err) std::fprintf(stderr, "failed\n");
		else std::fprintf(stderr, "passed\n");
	} else std::fprintf(stderr, "skipped\n");
	std::free(delta); std::free(g0);
}
