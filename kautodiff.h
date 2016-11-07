#ifndef KANN_AUTODIFF_H
#define KANN_AUTODIFF_H

#include <stdio.h>

#define KAD_ALLOC      1
#define KAD_FORWARD    2
#define KAD_BACKWARD   3
#define KAD_SYNC_DIM   4

#define KAD_MAX_DIM    4

struct kad_node_t;

typedef struct {
	struct kad_node_t *p; // child node
	float *t;             // temporary data needed for backprop, if not NULL; allocated
} kad_edge_t;

typedef struct kad_node_t {
	short n_d;            // number of dimensions; no larger than KAD_MAX_DIM
	short op;             // operator; kad_op_list[op] is the actual function
	int label;            // label for external uses
	int tmp;              // temporary field; MUST BE zero before calling kad_compile()
	short n_child;        // number of child nodes
	short to_back;        // whether to do back propogation
	int d[KAD_MAX_DIM];   // dimensions
	union {               
		const float *cx; 
		float *x;         // allocated for internal nodes
	} _;                  
	float *g;             // gradient; allocated for internal nodes
	kad_edge_t *child;    // child nodes
	void *ptr;
} kad_node_t;

typedef struct {
	void *data;
	double (*func)(void*);
} kad_rng_t;

typedef int (*kad_op_f)(kad_node_t*, int);
extern kad_op_f kad_op_list[];

#define kad_for1(p) (kad_op_list[(p)->op]((p), KAD_FORWARD))
#define kad_is_var(p) ((p)->n_child == 0 && (p)->to_back)

#ifdef __cplusplus
extern "C" {
#endif

kad_node_t *kad_par(const float *x, int n_d, ...);
kad_node_t *kad_var(const float *x, float *g, int n_d, ...);

kad_node_t *kad_add(kad_node_t *x, kad_node_t *y);   // z(x,y) = x + y (element-wise addition)
kad_node_t *kad_mul(kad_node_t *x, kad_node_t *y);   // z(x,y) = x * y (element-wise product)
kad_node_t *kad_cmul(kad_node_t *x, kad_node_t *y);  // z(x,y) = x * y^T (matrix product, with y transposed)
kad_node_t *kad_ce2(kad_node_t *x, kad_node_t *y);   // z(x,y) = \sum_i -y_i*log(f(x_i)) - (1-y_i)*log(1-f(x_i)); f() is sigmoid (binary cross-entropy for sigmoid; only x differentiable)

kad_node_t *kad_norm2(kad_node_t *x); // z(x) = \sum_i x_i^2 (L2 norm)
kad_node_t *kad_sigm(kad_node_t *x);  // z(x) = 1/(1+exp(-x)) (element-wise sigmoid)
kad_node_t *kad_tanh(kad_node_t *x);  // z(x) = (1-exp(-2x)) / (1+exp(-2x)) (element-wise tanh)
kad_node_t *kad_relu(kad_node_t *x);  // z(x) = max{0,x} (element-wise rectifier (aka ReLU))

kad_node_t **kad_compile(int *n_node, int n_roots, ...);
float kad_eval(int n, kad_node_t **a, int from, int cal_grad);
void kad_free_node(kad_node_t *p);
void kad_free(int n, kad_node_t **a);

void kad_write1(FILE *fp, const kad_node_t *p);
kad_node_t *kad_read1(FILE *fp, kad_node_t **node);
int kad_write(FILE *fp, int n_node, kad_node_t **node);
kad_node_t **kad_read(FILE *fp, int *_n_node);

float kad_sdot(int n, const float *x, const float *y);
void kad_saxpy(int n, float a, const float *x, float *y);

void kad_debug(FILE *fp, int n, kad_node_t **v);
void kad_check_grad(int n, kad_node_t **a, int from);

#ifdef __cplusplus
}
#endif

static inline int kad_len(const kad_node_t *p)
{
	int n = 1, i;
	for (i = 0; i < p->n_d; ++i) n *= p->d[i];
	return n;
}

static inline int kad_n_var(int n, kad_node_t *const* v)
{
	int c = 0, i;
	for (i = 0; i < n; ++i)
		if (v[i]->n_child == 0 && v[i]->to_back)
			c += kad_len(v[i]);
	return c;
}

#endif
