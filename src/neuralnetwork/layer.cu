#include "../includes.h"
#include "Network.h"

void fully_connected_layer::init(layer_data data, layer_data data_previous, float* new_delta) {

    data.n_in = {data_previous.n_out.feature_maps * data_previous.n_out.y * data_previous.n_out.x, 1, 1};
    data.elems = data.n_in.x+data_previous.elems;
    this->data = data;

    cudaMalloc((void**) &delta, data.n_out.x*sizeof(float));
    this->new_delta = new_delta;

    weights_size = data.n_out.x*data.n_in.x;
    biases_size = data.n_out.x;

    cudaMalloc((void**) &this->dev_data, sizeof(layer_data));
    cudaMalloc((void**) &this->dev_data_previous, sizeof(layer_data));

    cudaMemcpy(this->dev_data, &data, sizeof(layer_data), cudaMemcpyHostToDevice);
    cudaMemcpy(this->dev_data_previous, &data_previous, sizeof(layer_data), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &dev_weights, weights_size*sizeof(float));
    cudaMalloc((void**) &dev_weights_vel, weights_size*sizeof(float));
    cudaMalloc((void**) &dev_weights_updt, weights_size*sizeof(float));

    // weights init: https://www.analyticsvidhya.com/blog/2021/05/how-to-initialize-weights-in-neural-networks/
    // https://wandb.ai/sauravmaheshkar/initialization/reports/A-Gentle-Introduction-To-Weight-Initialization-for-Neural-Networks--Vmlldzo2ODExMTg
    // https://stats.stackexchange.com/questions/373136/softmax-weights-initialization
    float stddev;
    if (data.activation_function == RELU || data.activation_function == LEAKY_RELU) stddev = sqrt(2.0/data.n_in.x); // He-et-al
    else stddev = sqrt(2.0/data.n_in.x+data.n_out.x); // Xavier
    float* dev_stddev;
    cudaMalloc((void**) &dev_stddev, sizeof(float));
    cudaMemcpy(dev_stddev, &stddev, sizeof(float), cudaMemcpyHostToDevice);
    set_to_random<<<weights_size, 1>>>(dev_weights, dev_stddev);
    set_to<<<weights_size, 1>>>(dev_weights_vel, 0);
    set_to<<<weights_size, 1>>>(dev_weights_updt, 0);

    cudaMalloc((void**) &dev_biases, biases_size*sizeof(float));
    cudaMalloc((void**) &dev_biases_vel, biases_size*sizeof(float));
    cudaMalloc((void**) &dev_biases_updt, biases_size*sizeof(float));
    // biases init: https://medium.com/@glenmeyerowitz/bias-initialization-in-a-neural-network-2e5d26fed0f0
    set_to<<<biases_size,1>>>(dev_biases, 0.01);
    set_to<<<biases_size,1>>>(dev_biases_vel, 0);
    set_to<<<biases_size,1>>>(dev_biases_updt, 0);

    cudaFree(dev_stddev);
}

void fully_connected_layer::feedforward(float* dev_a, float* dev_dz) {
/*
    if (data.activation_function == SOFTMAX) {
        // TODO: make this smart
        reduce<<<data.n_out.x, data.n_in.x, data.n_in.x*sizeof(float)>>>(dev_weights, &dev_a[data.elems], &dev_data->n_in.x, &dev_data->n_in.x, CALC_Z, &dev_a[data.elems-data.n_in.x], dev_biases);
        cudaDeviceSynchronize();

        float* exp_vec;
        float* sum_of_exp;
        cudaMalloc((void**) &exp_vec, data.n_out.x*sizeof(float));
        cudaMalloc((void**) &sum_of_exp, sizeof(float));
        set_to<<<1,1>>> (sum_of_exp, 0);
        cudaDeviceSynchronize();
        //assert(data.n_out.x < (1<<10));

        int *max_id;
        cudaMalloc((void**) &max_id, sizeof(int));
        find_max<<<1,1>>>(&dev_a[data.elems], max_id, &dev_data->n_out.x);
        calc_exp<<<data.n_out.x, 1>>>(exp_vec, &dev_a[data.elems], max_id); // this could also be done in the reduce func
        cudaDeviceSynchronize();

        reduce<<<1, data.n_out.x, data.n_out.x*sizeof(float)>>>(exp_vec, sum_of_exp, &dev_data->n_out.x, &dev_data->n_out.x, ADD_EXP);
        cudaDeviceSynchronize();

        calc_a_and_dz<<<data.n_out.x, 1>>>(&dev_a[data.elems], &dev_dz[data.elems], &dev_data->activation_function, sum_of_exp);
        cudaDeviceSynchronize();

        cudaFree(max_id);
        cudaFree(exp_vec);
        cudaFree(sum_of_exp);
        //cudaDeviceSynchronize();
    } else {*/
        //reduce<<<data.n_out.x, data.n_in.x, data.n_in.x*sizeof(float)>>>(dev_weights, &dev_a[data.elems], &dev_data->n_in.x, &dev_data->n_in.x, CALC_Z, &dev_a[data.elems-data.n_in.x], dev_biases, &dev_dz[data.elems], &dev_data->activation_function
        dev_feedforward<<<data.n_out.x, data.n_in.x, data.n_in.x*sizeof(float)>>>(dev_weights, &dev_a[data.elems], &dev_data->n_in, &dev_a[data.elems-data.n_in.x], dev_biases, &dev_dz[data.elems], &dev_data->activation_function);
    cudaDeviceSynchronize();
    //}
}

void fully_connected_layer::backprop(float* activations, float* derivative_z) {
    backprop_update_w_b_fc<<<data.n_out.x, data.n_in.x>>>(dev_weights_updt, delta,
                                                          &activations[data.elems - data.n_in.x],
                                                          dev_biases_updt, &dev_data->n_in.x);
    dev_backprop<<<data.n_in.x, data.n_out.x, data.n_out.x * sizeof(float)>>>(delta,
                                                                              &derivative_z[data.elems - data.n_in.x],
                                                                              new_delta, dev_weights, &dev_data->n_in);
    cudaDeviceSynchronize();
}

void fully_connected_layer::update(hyperparams* dev_params) {
    // update velocities
    ::update<<<data.n_out.x, data.n_in.x>>> (dev_biases_vel, dev_weights_vel, dev_weights_updt, dev_biases_updt, dev_weights, dev_biases, dev_params);
    cudaDeviceSynchronize();
}

void fully_connected_layer::save(std::string filename) {
    std::ofstream file(filename, std::ios_base::app);

    file << LAYER_NUM_FULLY_CONNECTED << DEL;
    file << data.activation_function << DEL;
    file << data.n_out.x << DEL;
    file << biases_size << DEL;
    file << weights_size << DEL;

    float* biases = new float [data.n_out.x];
    cudaMemcpy(biases, dev_biases, biases_size*sizeof(float), cudaMemcpyDeviceToHost);
    for (int bias = 0; bias < biases_size; bias++) file << biases[bias] << " ";
    delete[] biases;
    file << DEL;

    float* biases_vel = new float [data.n_out.x];
    cudaMemcpy(biases_vel, dev_biases_vel, biases_size*sizeof(float), cudaMemcpyDeviceToHost);
    for (int bias_vel = 0; bias_vel < biases_size; bias_vel++) file << biases_vel[bias_vel] << " ";
    delete[] biases_vel;
    file << DEL;

    float* weights = new float [data.n_out.x*data.n_in.x];
    cudaMemcpy(weights, dev_weights, weights_size*sizeof(float), cudaMemcpyDeviceToHost);
    for (int weight = 0; weight < weights_size; weight++) file << weights[weight] << " ";
    delete[] weights;
    file << DEL;

    float* weights_vel = new float [data.n_out.x*data.n_in.x];
    cudaMemcpy(weights_vel, dev_weights_vel, weights_size*sizeof(float), cudaMemcpyDeviceToHost);
    for (int weight = 0; weight < weights_size; weight++) file << weights_vel[weight] << " ";
    delete[] weights_vel;
    file << "\n";

    file.close();
}

void fully_connected_layer::load(std::string line, layer_data *layer, float* &biases, float* &biases_vel, float* &weights, float* &weights_vel) {
    // TODO this could surely be written nicer
    std::stringstream ss_line(line);
    std::string str;
    getline(ss_line, str, DEL); // type

    getline(ss_line, str, DEL);
    layer->activation_function = atoi(str.c_str());

    getline(ss_line, str, DEL);
    layer->n_out.x = atoi(str.c_str());
    layer->n_out.y = 1;
    layer->n_out.feature_maps = 1;

    getline(ss_line, str, DEL);
    int biases_size = atoi(str.c_str());

    getline(ss_line, str, DEL);
    int weights_size = atoi(str.c_str());

    biases = new float[biases_size];
    biases_vel = new float[biases_size];
    weights = new float[weights_size];
    weights_vel = new float[weights_size];

    getline(ss_line, str, DEL);
    std::stringstream ss_str(str);
    for (int bias = 0; bias < biases_size; bias++) {
        std::string val_str;
        getline(ss_str, val_str, ' ');
        biases[bias] = atof(val_str.c_str());
    }

    getline(ss_line, str, DEL);
    ss_str = std::stringstream(str);
    for (int bias = 0; bias < biases_size; bias++) {
        std::string val_str;
        getline(ss_str, val_str, ' ');
        biases_vel[bias] = atof(val_str.c_str());
    }

    getline(ss_line, str, DEL);
    ss_str = std::stringstream(str);
    for (int weight = 0; weight < weights_size; weight++) {
        std::string val_str;
        getline(ss_str, val_str, ' ');
        weights[weight] = atof(val_str.c_str());
    }

    getline(ss_line, str, DEL);
    ss_str = std::stringstream(str);
    for (int weight = 0; weight < weights_size; weight++) {
        std::string val_str;
        getline(ss_str, val_str, ' ');
        weights_vel[weight] = atof(val_str.c_str());
    }
}

void fully_connected_layer::clear() {
    cudaFree(delta);
    cudaFree(dev_weights);
    cudaFree(dev_weights_vel);
    cudaFree(dev_weights_updt);
    cudaFree(dev_biases);
    cudaFree(dev_biases_vel);
    cudaFree(dev_biases_updt);
    cudaFree(dev_data_previous);
    cudaFree(dev_data);
}

void convolutional_layer::init(layer_data data, layer_data data_previous, float* new_delta) {

    data.n_in = data_previous.n_out;
    data.n_out.x = (data.n_in.x - data.receptive_field_length + 1) / data.stride_length;
    data.n_out.y = (data.n_in.y - data.receptive_field_length + 1) / data.stride_length;
    data.elems = data.n_in.x*data.n_in.y*data.n_in.feature_maps+data_previous.elems;
    this->data = data;

    weights_size = data.n_in.feature_maps * data.n_out.feature_maps * data.receptive_field_length * data.receptive_field_length;
    biases_size = data.n_out.feature_maps;

    cudaMalloc((void**) &delta, data.n_out.x*data.n_out.y*data.n_out.feature_maps*sizeof(float)); // TODO SIZE?
    this->new_delta = new_delta;

    cudaMalloc((void**) &this->dev_data, sizeof(layer_data));
    cudaMalloc((void**) &this->dev_data_previous, sizeof(layer_data));

    cudaMemcpy(this->dev_data, &data, sizeof(layer_data), cudaMemcpyHostToDevice);
    cudaMemcpy(this->dev_data_previous, &data_previous, sizeof(layer_data), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &dev_weights, weights_size*sizeof(float));
    cudaMalloc((void**) &dev_weights_vel, weights_size*sizeof(float));
    cudaMalloc((void**) &dev_weights_updt, weights_size*sizeof(float));
    // weights init: https://www.analyticsvidhya.com/blog/2021/05/how-to-initialize-weights-in-neural-networks/
    // https://wandb.ai/sauravmaheshkar/initialization/reports/A-Gentle-Introduction-To-Weight-Initialization-for-Neural-Networks--Vmlldzo2ODExMTg
    // https://stats.stackexchange.com/questions/373136/softmax-weights-initialization
    float stddev;
    // TODO He-et-al convolutional
    if (data.activation_function == RELU) stddev = sqrt(2.0/(data.n_in.x*data.n_in.y*data.n_in.feature_maps)); // He-et-al
    else stddev = sqrt(2.0/(data.n_in.x*data.n_in.y*data.n_in.feature_maps+(data.n_out.x*data.n_out.y*data.n_out.feature_maps))); // Xavier

    float* dev_stddev;
    cudaMalloc((void**) &dev_stddev, sizeof(float));
    cudaMemcpy(dev_stddev, &stddev, sizeof(float), cudaMemcpyHostToDevice);
    set_to_random<<<weights_size, 1>>>(dev_weights, dev_stddev);
    set_to<<<weights_size, 1>>>(dev_weights_vel, 0);
    set_to<<<weights_size, 1>>>(dev_weights_updt, 0);

    cudaMalloc((void**) &dev_biases, biases_size*sizeof(float));
    cudaMalloc((void**) &dev_biases_vel, biases_size*sizeof(float));
    cudaMalloc((void**) &dev_biases_updt, biases_size*sizeof(float));
    // biases init: https://medium.com/@glenmeyerowitz/bias-initialization-in-a-neural-network-2e5d26fed0f0
    set_to<<<biases_size, 1>>>(dev_biases, 0.01);
    set_to<<<biases_size,1>>>(dev_biases_vel, 0);
    set_to<<<biases_size,1>>>(dev_biases_updt, 0);

    cudaFree(dev_stddev);

    cudaDeviceSynchronize();
}

void convolutional_layer::feedforward(float* dev_a, float* dev_dz) {
    dim3 blocks(data.n_out.x, data.n_out.y, data.n_out.feature_maps);
    dim3 threads(data.receptive_field_length, data.receptive_field_length, data.n_in.feature_maps);
    int previous_elems = data.elems - (data.n_in.x*data.n_in.y*data.n_in.feature_maps);
    dev_feedforward<<<blocks, threads, data.receptive_field_length*data.receptive_field_length*data.n_in.feature_maps*sizeof(float)>>>(dev_weights, &dev_a[data.elems], &dev_data->n_in, &dev_a[previous_elems], dev_biases, &dev_dz[data.elems], &dev_data->activation_function, &dev_data->stride_length);
    cudaDeviceSynchronize();
}

void convolutional_layer::backprop(float* activations, float* derivative_z) {
    dim3 blocks(data.receptive_field_length, data.receptive_field_length, data.n_in.feature_maps*data.n_out.feature_maps);
    dim3 threads(data.n_out.x, data.n_out.y);
    backprop_update_w_b_conv<<<blocks, threads, data.n_out.x*data.n_out.y*sizeof(float)>>>(dev_weights_updt, delta,
                                                          &activations[data.elems - data.n_in.x],
                                                          dev_biases_updt, &dev_data->n_in, &dev_data->stride_length);

    blocks = dim3(data.n_in.x, data.n_in.y, data.n_in.feature_maps);
    threads = dim3(data.receptive_field_length, data.receptive_field_length, data.n_out.feature_maps);

    dev_backprop<<<blocks, threads, data.n_out.x * data.n_out.y * data.n_out.feature_maps * sizeof(float)>>>(delta,
                                                                              &derivative_z[data.elems - data.n_in.x],
                                                                              new_delta, dev_weights, &dev_data->n_out, &dev_data->stride_length);

    cudaDeviceSynchronize();
}

void convolutional_layer::update(hyperparams* dev_params) {
    ::update<<<data.n_out.feature_maps, data.n_in.feature_maps*data.receptive_field_length*data.receptive_field_length>>> (dev_biases_vel, dev_weights_vel, dev_weights_updt, dev_biases_updt, dev_weights, dev_biases, dev_params, &dev_data->stride_length, &dev_data->n_out);
    cudaDeviceSynchronize();
}

void convolutional_layer::save(std::string filename) {
    std::ofstream file(filename, std::ios_base::app);

    file << LAYER_NUM_CONVOLUTIONAL << DEL;
    file << data.activation_function << DEL;
    file << data.stride_length << DEL << data.receptive_field_length << DEL << data.n_out.feature_maps << DEL;
    file << biases_size << DEL;
    file << weights_size << DEL;

    float* biases = new float [data.n_out.feature_maps];
    cudaMemcpy(biases, dev_biases, data.n_out.feature_maps*sizeof(float), cudaMemcpyDeviceToHost);
    for (int bias = 0; bias < data.n_out.feature_maps; bias++) file << biases[bias] << " ";
    delete[] biases;
    file << DEL;

    float* biases_vel = new float [data.n_out.feature_maps];
    cudaMemcpy(biases_vel, dev_biases_vel, data.n_out.feature_maps*sizeof(float), cudaMemcpyDeviceToHost);
    for (int bias_vel = 0; bias_vel < data.n_out.feature_maps; bias_vel++) file << biases_vel[bias_vel] << " ";
    delete[] biases_vel;
    file << DEL;

    float* weights = new float [weights_size];
    cudaMemcpy(weights, dev_weights, weights_size*sizeof(float), cudaMemcpyDeviceToHost);
    for (int weight = 0; weight < weights_size; weight++) file << weights[weight] << " ";
    delete[] weights;
    file << DEL;

    float* weights_vel = new float [weights_size];
    cudaMemcpy(weights_vel, dev_weights, weights_size*sizeof(float), cudaMemcpyDeviceToHost);
    for (int weight_vel = 0; weight_vel < weights_size; weight_vel++) file << weights_vel[weight_vel] << " ";
    delete[] weights_vel;
    file << "\n";

    file.close();
}

void convolutional_layer::load(std::string line, layer_data *layer, float* &biases, float* &biases_vel, float* &weights, float* &weights_vel) {
    // TODO this could surely be written nicer
    std::stringstream ss_line(line);
    std::string str;
    getline(ss_line, str, DEL); // type

    getline(ss_line, str, DEL);
    layer->activation_function = atoi(str.c_str());

    getline(ss_line, str, DEL);
    layer->stride_length = atoi(str.c_str());

    getline(ss_line, str, DEL);
    layer->receptive_field_length = atoi(str.c_str());

    getline(ss_line, str, DEL);
    layer->n_out.feature_maps = atoi(str.c_str());

    getline(ss_line, str, DEL);
    int biases_size = atoi(str.c_str());

    getline(ss_line, str, DEL);
    int weights_size = atoi(str.c_str());

    biases = new float[biases_size];
    biases_vel = new float[biases_size];
    weights = new float[weights_size];
    weights_vel = new float[weights_size];

    getline(ss_line, str, DEL);
    std::stringstream ss_str(str);
    for (int bias = 0; bias < biases_size; bias++) {
        std::string val_str;
        getline(ss_str, val_str, ' ');
        biases[bias] = atof(val_str.c_str());
    }

    getline(ss_line, str, DEL);
    ss_str = std::stringstream(str);
    for (int bias = 0; bias < biases_size; bias++) {
        std::string val_str;
        getline(ss_str, val_str, ' ');
        biases_vel[bias] = atof(val_str.c_str());
    }

    getline(ss_line, str, DEL);
    ss_str = std::stringstream(str);
    for (int weight = 0; weight < weights_size; weight++) {
        std::string val_str;
        getline(ss_str, val_str, ' ');
        weights[weight] = atof(val_str.c_str());
    }

    getline(ss_line, str, DEL);
    ss_str = std::stringstream(str);
    for (int weight = 0; weight < weights_size; weight++) {
        std::string val_str;
        getline(ss_str, val_str, ' ');
        weights_vel[weight] = atof(val_str.c_str());
    }
}

void convolutional_layer::clear() {
    cudaFree(delta);
    cudaFree(dev_weights);
    cudaFree(dev_weights_vel);
    cudaFree(dev_weights_updt);
    cudaFree(dev_biases);
    cudaFree(dev_biases_vel);
    cudaFree(dev_biases_updt);
    cudaFree(dev_data_previous);
    cudaFree(dev_data);
}

void input_layer::init(layer_data data, layer_data data_previous, float* new_delta) {
    data.elems = 0;
    this->data = data;
    cudaMalloc((void**) &delta, data.n_out.feature_maps*data.n_out.y*data.n_out.x*sizeof(float));

    biases_size = 0;
    weights_size = 0;
    (void) data_previous;
}

void input_layer::feedforward(float* a, float* dz) {}

void input_layer::backprop(float* activations, float* derivative_z) {}

void input_layer::update(hyperparams* params) {}

void input_layer::save(std::string filename) {
    std::ofstream file(filename, std::ios_base::app);

    file << LAYER_NUM_INPUT << DEL;
    file << data.n_out.x << DEL << data.n_out.y << "\n";

    file.close();
}

void input_layer::load(std::string line, layer_data *layer, float* &biases, float* &biases_vel, float* &weights, float* &weights_vel) {
    std::stringstream ss_line(line);
    std::string str;
    getline(ss_line, str, DEL); // type

    getline(ss_line, str, DEL);
    layer->n_out.x = atoi(str.c_str());

    getline(ss_line, str, DEL);
    layer->n_out.y = atoi(str.c_str());

    layer->n_out.feature_maps = 1;
}

void input_layer::clear() {
    cudaFree(delta);
}