#ifndef _BACKPROP_H_
#define _BACKPROP_H_

#define BIGRND 0x7fffffff

#define GPU
#define THREADS 256
#define WIDTH 16  // shared memory width  
#define HEIGHT 16 // shared memory height

#define ETA 0.3       //eta value
#define MOMENTUM 0.3  //momentum value
#define NUM_THREAD 4  //OpenMP threads


typedef struct {
  int input_n;                  /* number of input units */
  int hidden_n;                 /* number of hidden units */
  int output_n;                 /* number of output units */

  float *input_units;          /* the input units */
  float *hidden_units;         /* the hidden units */
  float *output_units;         /* the output units */

  float *hidden_delta;         /* storage for hidden unit error */
  float *output_delta;         /* storage for output unit error */

  float *target;               /* storage for target vector */

  float **input_weights;       /* weights from input to hidden layer */
  float **hidden_weights;      /* weights from hidden to output layer */

                                /*** The next two are for momentum ***/
  float **input_prev_weights;  /* previous change on input to hidden wgt */
  float **hidden_prev_weights; /* previous change on hidden to output wgt */
} BPNN;

#ifdef __cplusplus
extern "C" {
#endif
int setup(int argc, char** argv);
#ifdef __cplusplus
}
#endif

/*** User-level functions ***/

void backprop_face(void);

void bpnn_initialize(int seed);

BPNN *bpnn_create(int layer_size, int n_hidden, int n_out);
void bpnn_free(BPNN* net);

void bpnn_train(BPNN* net, float* eo, float* eh);

#ifdef __cplusplus
extern "C" {
#endif
void bpnn_train_cuda(BPNN* net, float* eo, float* eh);
#ifdef __cplusplus
}
#endif
void bpnn_feedforward(BPNN* net);

void bpnn_save(BPNN* net, char* filename);
//BPNN *bpnn_read(char* filename);

void load(BPNN* net);

#ifdef __cplusplus
extern "C" {
#endif
float squash(float x);
float* alloc_1d_dbl(int n);
float** alloc_2d_dbl(int m, int n);
void bpnn_randomize_weights(float** w, int m, int n);
void bpnn_randomize_row(float* w, int m);
void bpnn_zero_weights(float** w, int m, int n);
BPNN* bpnn_internal_create(int n_in, int n_hidden, int n_out);
void bpnn_layerforward(float* l1, float* l2, float** conn, int n1, int n2);
void bpnn_output_error(float* delta, float* target, float* output, int nj, float* err);
void bpnn_hidden_error(float* delta_h, int nh, float* delta_o, int no,
        float** who, float* hidden, float* err);
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly,
        float **w, float **oldw);
#ifdef __cplusplus
}
#endif

#endif
