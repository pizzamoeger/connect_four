#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

extern int* zero_pointer;
extern float* f_zero_pointer;

struct hyperparams {
    float convolutional_weights_learning_rate;
    float convolutional_biases_learning_rate;
    float fully_connected_weights_learning_rate;
    float fully_connected_biases_learning_rate;
    float convWRed;
    float convBRed;
    float fcWRed;
    float fcBRed;
    float L2_regularization_term;
    float momentum_coefficient;
    int epochs;
    int mini_batch_size;
    int training_data_size;
    int test_data_size;
    int cost;
};

enum {
    LAYER_NUM_FULLY_CONNECTED,
    LAYER_NUM_CONVOLUTIONAL,
    LAYER_NUM_INPUT
};

enum {
    SIGMOID,
    RELU,
    SOFTMAX,
    TANH,
    LEAKY_RELU,
    NONE
};

enum {
    CROSSENTROPY,
    MSE
};

#define OUTPUT_NEURONS 7
#define INPUT_NEURONS_X 6
#define INPUT_NEURONS_Y 7
#define INPUT_NEURONS INPUT_NEURONS_X*INPUT_NEURONS_Y
#define DEL '/'

// FIND-TAG-N
#define NEURONS 510

inline __device__ float activation_function(float x, int activation_func, float sum_of_exp);
inline __device__ float activation_function_prime(float x, int activation_func, float sum_of_exp);

std::pair<std::vector<std::pair<float*,float*>>, int> load_data(std::string filename); // TODO : watch this https://www.youtube.com/watch?v=m7E9piHcfr4 to make this faster
hyperparams get_params();
void clear_data(std::vector<std::pair<float*,float*>> & data);

struct network_data {
    int x;
    int y;
    int feature_maps;
};

struct layer_data {
    int type;

    network_data n_in;
    network_data n_out;

    int elems;

    int stride_length;
    int receptive_field_length;

    int activation_function;
};

struct layer {
    layer_data data;
    layer_data* dev_data;
    float* delta;
    float* new_delta;

    int weights_size;
    int biases_size;

    float* dev_biases;
    float* dev_biases_vel;
    float* dev_biases_updt;

    float* dev_weights;
    float* dev_weights_vel;
    float* dev_weights_updt;

    virtual void init(layer_data data, layer_data data_previous, float* new_delta) = 0;
    virtual void feedforward(float* a, float* dz) = 0;
    virtual void backprop(float* activations, float* derivative_z) = 0;
    virtual void update(hyperparams* params) = 0;
    virtual void save(std::string file) = 0;
    virtual void load(std::string line, layer_data* layer, float* &biases, float* &biases_vel, float* &weights, float* &weights_vel) = 0;
    virtual void clear() = 0;
};

struct fully_connected_layer : public layer {
    layer_data* dev_data_previous;

    void init (layer_data data, layer_data data_previous, float* new_delta);

    void feedforward(float* a, float* dz);

    void backprop(float* activations, float* derivative_z);

    void update(hyperparams* params);

    void save(std::string filename);

    void load(std::string line, layer_data* layer, float* &biases, float* &biases_vel, float* &weights, float* &weights_vel);

    void clear();
};

struct convolutional_layer : public layer {
    layer_data* dev_data_previous;

    void init (layer_data data, layer_data data_previous, float* new_delta);

    void feedforward(float* a, float* dz);

    void backprop(float* activations, float* derivative_z);

    void update(hyperparams* params);

    void save(std::string filename);

    void load(std::string line, layer_data* layer, float* &biases, float* &biases_vel, float* &weights, float* &weight_vel);

    void clear();
};

struct input_layer : public layer {

    void init (layer_data data, layer_data data_previous, float* new_delta);

    void feedforward(float* a, float* dz);

    void backprop(float* activations, float* derivative_z);

    void update(hyperparams* params);

    void save(std::string filename);

    void load(std::string line, layer_data* layer, float* &biases, float* &biases_vel, float* &weights, float* &weights_vel);

    void clear();
};

struct Network {
    // number of layers
    int L;

    float* activations;
    float* derivatives_z;

    // layers
    std::unique_ptr<layer> *layers;

    hyperparams params;
    hyperparams* dev_params;

    Network();

    void init (layer_data* layers, int L, hyperparams params);

    void feedforward(float* a, float* activations, float* dz);

    void SGD(std::vector<std::pair<float*,float*>> training_data, std::vector<std::pair<float*,float*>> test_data);

    void update_mini_batch(std::vector<std::pair<float*,float*>> mini_batch);

    void backprop(float* in, float* out);

    void save(std::string filename);

    void load(std::string filename, hyperparams params, layer_data* &layers);

    void clear();

    std::pair<int,int> evaluate(std::vector<std::pair<float*,float*>> test_data, int test_data_size);
};

int get_convolutional_weights_index(int previous_map, int map, int y, int x, layer_data &data);
int get_data_index(int map, int y, int x, layer_data &data);
inline __device__ int get_fully_connected_weight_index_dev (int neuron, int previous_neuron, int data_n_in);

__global__ void set_delta (float* delta, float* activations, float* out, int* cost_func);
__global__ void update (float* biases_vel, float* weights_vel, float* weights_updt, float* biases_updt, float* weights, float* biases, hyperparams* params, int* stride_length = NULL, network_data* n_out = NULL);
__global__ void eval (float* correct, float* output, int* counter, int* size);

__global__ void set_to (float *vec, float value); // initialize the elements to value
__global__ void set_to_random (float *vec, float* stddev); // initialize the elements to random value with mean 0 and given stddev

inline __device__ void reduce_last_warp(volatile float* sum, int ind, int block_size);
inline __device__ void reduce(int tid, int block_size, volatile float* sum);
__global__ void dev_feedforward(float* weights, float* new_a, network_data* n_in, float* a, float* biases, float* new_dz, int* activation_function, int* stride_length = NULL);
__global__ void backprop_update_w_b_fc (float* dev_weights_upt, float* dev_delta, float* dev_activations, float* dev_biases_updt, int* data_n_in_x);
__global__ void backprop_update_w_b_conv (float* dev_weights_upt, float* dev_delta, float* dev_activations, float* dev_biases_updt, network_data* n_in, int* stride_len);
__global__ void dev_backprop(float* delta, float* dz, float* new_delta, float* weights, network_data* n, int* stride_len = NULL);

#endif
