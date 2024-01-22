#include "../includes.h"
#include "Network.h"

// sigmoid function and its derivative
inline __device__ float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

inline __device__ float sigmoid_prime(float x) {
    return (sigmoid(x)*(1-sigmoid(x)));
}

inline __device__ float relu(float x){
    return max(x, 0.0f);
}

inline __device__ float relu_prime(float x){
    if (x > 0) return 1.0f;
    return 0.0f;
}

inline __device__ float leaky_relu(float x){
    float ret;
    if (x > 0) ret = x;
    else ret = x*0.1; // TODO store negative slope somewhere
    return ret;
}

inline __device__ float leaky_relu_prime(float x){
    if (x > 0) return 1.0f;
    else return 0.1; // TODO store negative slope somewhere
}

inline __device__ float tanh_prime(float x){
    return 1 - tanh(x)*tanh(x);
}

inline __device__ float softmax(float x, float sum_of_exp) {
    return expf(x)/sum_of_exp;
}

inline __device__ float softmax_prime(float x, float sum_of_exp) {
    return (softmax(x, sum_of_exp)*(1-softmax(x, sum_of_exp)));
}

inline __device__ float cross_entropy_prime(float out_net, float out_cor) {
    return (out_net-out_cor);
}

inline __device__ float MSE_prime(float out_net, float out_cor) { // TODO this is incorrect
    return (out_net-out_cor)*(out_net-out_cor);
}

inline __device__ float activation_function(float x, int activation_func, float sum_of_exp) {
    switch (activation_func) {
        case SIGMOID:
            return sigmoid(x);
        case RELU:
            return relu(x);
        case SOFTMAX:
            return softmax(x, sum_of_exp);
        case TANH:
            return tanh(x);
        case LEAKY_RELU:
            return leaky_relu(x);
        default:
            return x;
    }
}

inline __device__ float activation_function_prime(float x, int activation_func, float sum_of_exp) {
    switch (activation_func) {
        case SIGMOID:
            return sigmoid_prime(x);
        case RELU:
            return relu_prime(x);
        case SOFTMAX:
            return softmax_prime(x, sum_of_exp);
        case TANH:
            return tanh_prime(x);
        case LEAKY_RELU:
            return leaky_relu_prime(x);
        default:
            return 1;
    }
}

inline __device__ float cost_function_prime(float out_net, float out_cor, int cost_function) {
    if (cost_function == CROSSENTROPY) return cross_entropy_prime(out_net, out_cor);
    else if (cost_function == MSE) return MSE_prime(out_net, out_cor);
    else return 0;
}

inline __device__ int get_fully_connected_weight_index_dev (int neuron, int previous_neuron, int data_n_in) {
    return neuron*data_n_in+previous_neuron;
}

// load data
std::pair<std::vector<std::pair<float*,float*>>, int> load_data(std::string filename) {
    // loads data from csv file of form label, pixel1, pixel2, pixel3, ..., pixel784
    std::ifstream file;
    std::string line;

    file.open(filename);

    // how many lines there are in the file
    int dataPoints = 0;
    while (getline(file, line)) {
        dataPoints++;
    }

    file.clear(); // Reset stream state
    file.seekg(0); // Move cursor back to beginning

    int lineIndex = 0;
    std::vector<std::pair<float*,float*>> data (dataPoints, {nullptr, nullptr});

    while (getline(file, line)) {
        std::stringstream ss(line);
        float* data_in = new float [INPUT_NEURONS];
        float* data_out = new float [OUTPUT_NEURONS];

        for (int i = 0; i < INPUT_NEURONS; i++) data_in[i] = 0;
        for (int i = 0; i < OUTPUT_NEURONS; i++) data_out[i] = 0;

        int label = -1;
        int i = 0;
        while (ss.good()) {
            std::string substr;
            getline(ss, substr, ' ');
            if (label == -1) {
                label = stoi(substr);
            } else {
                if (i == INPUT_NEURONS) break;
                data_in[i] = atof(substr.c_str());
                i++;
            }
        }
        data_out[label] = 1;


        float* dev_data_in;
        float* dev_data_out;
        cudaMalloc((void**) &dev_data_in, INPUT_NEURONS*sizeof(float));
        cudaMalloc((void**) &dev_data_out, OUTPUT_NEURONS*sizeof(float));
        cudaMemcpy(dev_data_in, data_in, INPUT_NEURONS*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_data_out, data_out, OUTPUT_NEURONS*sizeof(float), cudaMemcpyHostToDevice);
        data[lineIndex] = {dev_data_in, dev_data_out};

        lineIndex++;

        delete [] data_in;
        delete [] data_out;
    }

    std::cerr << dataPoints << " data loaded from " + filename + "\n";
    file.close();
    return {data, dataPoints};
}

void clear_data(std::vector<std::pair<float*,float*>> & data) {
    for (int data_point = 0; data_point < (int)data.size(); data_point++) {
        cudaFree(data[data_point].first);
        cudaFree(data[data_point].second);
    }
}

__global__ void set_delta (float* delta, float* activations, float* out, int* cost_func) {
    int neuron = blockIdx.x;
    delta[neuron] = cost_function_prime(activations[neuron], out[neuron], *cost_func);
}

__global__ void backprop_update_w_b_fc (float* dev_weights_upt, float* dev_delta, float* dev_activations, float* dev_biases_updt, int* data_n_in_x) {
    int neuron = blockIdx.x;
    int previous_neuron = threadIdx.x;
    dev_weights_upt[get_fully_connected_weight_index_dev(neuron, previous_neuron, *data_n_in_x)] += dev_delta[neuron] * dev_activations[previous_neuron];

    if (previous_neuron == 0) dev_biases_updt[neuron] += dev_delta[neuron];
}

__global__ void update (float* biases_vel, float* weights_vel, float* weights_updt, float* biases_updt, float* weights, float* biases, hyperparams* params, int* stride_length, network_data* n_out) {
    int neuron = blockIdx.x;
    int previous_neuron = threadIdx.x;
    int weight = neuron*blockDim.x+previous_neuron;

    if (previous_neuron == 0) {
        if (stride_length != NULL) { // conv
            biases_vel[neuron] = params->momentum_coefficient * biases_vel[neuron] -
                                (params->convolutional_biases_learning_rate / params->mini_batch_size) *
                                biases_updt[neuron];
        } else { // fully connected
            biases_vel[neuron] = params->momentum_coefficient * biases_vel[neuron] -
                                 (params->fully_connected_biases_learning_rate / params->mini_batch_size) *
                                 biases_updt[neuron];
        }
        biases[neuron] += biases_vel[neuron];
        biases_updt[neuron] = 0;
    }

    if (stride_length != NULL) { // conv
        weights_vel[weight] =
                params->momentum_coefficient * weights_vel[weight] -
                (params->convolutional_weights_learning_rate / params->mini_batch_size /
                 (n_out->x * n_out->y) *
                 *stride_length * *stride_length) * weights_updt[weight];

        weights[weight] = (1 - params->convolutional_weights_learning_rate / (n_out->x * n_out->y) *
                        *stride_length * *stride_length * params->L2_regularization_term) *
                            weights[weight] + weights_vel[weight];
    } else { // fully connected
        weights_vel[weight] = params->momentum_coefficient * weights_vel[weight] -
                (params->fully_connected_weights_learning_rate / params->mini_batch_size) *
                weights_updt[weight];
        weights[weight] = (1 - params->fully_connected_weights_learning_rate * params->L2_regularization_term) * weights[weight] + weights_vel[weight];
    }

    weights_updt[weight] = 0;
}

__global__ void eval (float* correct, float* output, int* counter, int* size) {
    int index = 0;

    for (int i = 0; i < (*size); i++) {
        if (output[i] > output[index]) index = i;
    }

    if (correct[index] == 1) (*counter)++;
}

__global__ void set_to (float *vec, float value) {
    int index = blockIdx.x;
    vec[index] = value;
}

__global__ void set_to_random (float *vec, float *stddev) {
    int index = blockIdx.x;
    curandState state;
    curand_init(clock64(), index, 0, &state);
    vec[index] = curand_normal(&state)*(*stddev);
}

inline __device__ void reduce_last_warp(volatile float* sum, int ind, int block_size) {
    if (ind < block_size - 32 && ind < 32) {
        sum[ind] += sum[ind + 32];
    }
    if (ind < block_size - 16 && ind < 16) {
        sum[ind] += sum[ind + 16];
    }
    if (ind < block_size - 8 && ind < 8) {
        sum[ind] += sum[ind + 8];
    }
    if (ind < block_size - 4 && ind < 4) {
        sum[ind] += sum[ind + 4];
    }
    if (ind < block_size - 2 && ind < 2) {
        sum[ind] += sum[ind + 2];
    }
    if (ind < block_size - 1 && ind < 1) {
        sum[ind] += sum[ind + 1];
    }
}

inline __device__ void reduce(int tid, int block_size, volatile float* sum) {
    // TODO see if faster if syncthreads outside and biases update in conv has an extra call or like this where syncthreads is outside of if
    if (tid < block_size - 512) {
        sum[tid] += sum[tid + 512];
    }
    __syncthreads();
    if (tid < block_size - 256 && tid < 256) {
        sum[tid] += sum[tid + 256];
    }
    __syncthreads();
    if (tid < block_size - 128 && tid < 128) {
        sum[tid] += sum[tid + 128];
    }
    __syncthreads();
    if (tid < block_size - 64 && tid < 64) {
        sum[tid] += sum[tid + 64];
    }
    __syncthreads();

    if (tid < 32) reduce_last_warp(sum, tid, block_size);

}

__global__ void dev_feedforward(float* weights, float* new_a, network_data* n_in, float* a, float* biases, float* new_dz, int* activation_func, int* stride_length) {
    int previous_neuron = threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
    int neuron = blockIdx.z*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x;

    extern __shared__ float sum[];

    // assumes n is <= max block size
    int n = blockDim.x*blockDim.y*blockDim.z;
    if (stride_length != NULL) {
        int previous_neuron_a = threadIdx.z*n_in->x*n_in->y
                          + (blockIdx.y*(*stride_length)+threadIdx.y)*n_in->x
                          + (blockIdx.x*(*stride_length)+threadIdx.x);
        sum[previous_neuron] = weights[blockIdx.z*n + previous_neuron]*a[previous_neuron_a];
    } // convolutional
    else sum[previous_neuron] = weights[neuron*n + previous_neuron]*a[previous_neuron]; // fully connected

    __syncthreads();

    reduce(previous_neuron, n, sum);

    if (previous_neuron == 0) {
        int neuron_b = neuron;
        if (stride_length != NULL) neuron_b = blockIdx.z;
        sum[0] += biases[neuron_b];
        new_dz[neuron] = activation_function_prime(sum[0], *activation_func, 0);
        new_a[neuron] = activation_function(sum[0], *activation_func, 0);
    }
}

__global__ void dev_backprop(float* delta, float* dz, float* new_delta, float* weights, network_data* n, int* stride_len) {
    extern __shared__ float sum[];

    int neuron = threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
    int tid = neuron;
    int previous_neuron = blockIdx.z*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x;

    int n_in = gridDim.x*gridDim.y*gridDim.z;
    int n_out = blockDim.x*blockDim.y*blockDim.z;

    if (stride_len != NULL) {
        // convolutional
        int x = blockIdx.x-threadIdx.x;
        int y = blockIdx.y-threadIdx.y;
        neuron = threadIdx.x*n->x*n->y + y/(*stride_len)*n->x + x/(*stride_len);
        int weight = threadIdx.z*blockDim.x*blockDim.y*gridDim.z + blockIdx.z * blockDim.x*blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        if (x < 0 || y < 0 || x % *stride_len != 0 || y % *stride_len != 0) {
            sum[tid] = 0;
        }
        else sum[tid] = delta[neuron]*weights[weight];
    } else sum[tid] = delta[neuron]*weights[neuron*n_in + previous_neuron];

    __syncthreads();

    reduce(tid, n_out, sum);

    if (tid == 0) {
        new_delta[previous_neuron] = sum[tid]*dz[previous_neuron];
    }
}

__global__ void backprop_update_w_b_conv (float* dev_weights_upt, float* dev_delta, float* dev_activations, float* dev_biases_updt, network_data* n_in, int* stride_length) {
    int map = blockIdx.z / n_in->feature_maps;
    int prev_map = blockIdx.z % n_in->feature_maps;
    int kernel_x = blockIdx.x;
    int kernel_y = blockIdx.y;
    int x = threadIdx.x;
    int y = threadIdx.y;

    int previous_neuron = prev_map*n_in->y*n_in->x + (y*(*stride_length)+kernel_y)*n_in->x + x*(*stride_length)+kernel_x;
    int neuron = map*blockDim.y*blockDim.x + y*blockDim.x + x;
    int tid = y * blockDim.x + x;

    extern __shared__ float sum[];

    int n = blockDim.y*blockDim.x; // how much to reduce
    sum[tid] = dev_activations[previous_neuron] * dev_delta[neuron];

    __syncthreads();

    reduce(tid, n, sum);

    if (tid == 0) {
        int weight = map * n_in->feature_maps * gridDim.y * gridDim.x + prev_map * gridDim.y * gridDim.x + kernel_y * gridDim.x + kernel_x;
        dev_weights_upt[weight] += sum[tid];
    }

    if (prev_map != 0 || kernel_x != 0 || kernel_y != 0) return; // biases: only for each map

    sum[tid] = dev_delta[neuron];
    __syncthreads();

    reduce(tid, n, sum);
    if (tid == 0) {
        dev_biases_updt[map] += sum[tid];
    }
}

__global__ void get_dqn_out (float* main_output, float* tar_output, float* reward, int* action, float* output, float* discount_factor) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float output_goal;
    if (abs(*reward) == 1.0) output_goal = *reward; // game is won or illegal move
    else {
        // get max of target out
        float max_out = 0;
        for (int i = 0; i < gridDim.x; i++) {
            if (tar_output[i] > max_out || i == 0) max_out = tar_output[i];
        }

        output_goal = (*reward) - (*discount_factor) * max_out;
    }

    for (int i = 0; i < gridDim.x; i++) {
        if (i == (*action)) output[i] = output_goal;
        else output[i] = main_output[i];
    }

}