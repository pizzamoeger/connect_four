#include "../game.h"
#include "../neuralnetwork/Network.h"

void convert_argv(int argc, char** argv, DQN& player, int& games, hyperparams& params) {
    if(argc%2 != 1) {
        std::cerr << "Invalid commandline arguments. Please check README.md\n";
        exit(0);
    }

    for (int arg = 1; arg < argc; arg+=2) {
        char* key = argv[arg];
        std::string val = argv[arg+1];
        // DQN params
        if (strcmp(key, "c")==0) player.c = atoi(val.c_str());
        else if (strcmp(key, "batch_size")==0) player.batch_size = atoi(val.c_str());
        else if (strcmp(key, "games")==0) games = atoi(val.c_str());
        else if (strcmp(key, "replay_buffer_size")==0) player.replay_buffer_size = atoi(val.c_str());
        else if (strcmp(key, "discount_factor")==0) player.discount_factor = atof(val.c_str());
        else if (strcmp(key, "epsilon_red")==0) player.epsilon_red = atof(val.c_str());
        else if (strcmp(key, "epsilon")==0) player.epsilon = atof(val.c_str());
        // neural network params
        else if (strcmp(key, "fcw")==0) params.fully_connected_weights_learning_rate = atof(val.c_str());
        else if (strcmp(key, "fcb")==0) params.fully_connected_biases_learning_rate = atof(val.c_str());
        else if (strcmp(key, "cw")==0) params.convolutional_weights_learning_rate = atof(val.c_str());
        else if (strcmp(key, "cb")==0) params.convolutional_biases_learning_rate = atof(val.c_str());
        else if (strcmp(key, "fcwr")==0) params.fcWRed = atof(val.c_str());
        else if (strcmp(key, "fcbr")==0) params.fcBRed = atof(val.c_str());
        else if (strcmp(key, "cwr")==0) params.convWRed = atof(val.c_str());
        else if (strcmp(key, "cbr")==0) params.convBRed = atof(val.c_str());
        else if (strcmp(key, "L2")==0) params.L2_regularization_term = atof(val.c_str());
        else if (strcmp(key, "momcoef")==0) params.momentum_coefficient = atof(val.c_str());
        else {
            std::cout << key << ": no valid key. Please check README.md\n";
            exit(0);
        }
    }
}

int main(int argc, char** argv) {

    // randomness
    srand(time(NULL));

    DQN player;

    // design layers
    layer_data input_layer;
    input_layer.type = LAYER_NUM_INPUT;
    input_layer.n_out = {INPUT_NEURONS_H, INPUT_NEURONS_W, 1};

    layer_data output_layer;
    output_layer.type = LAYER_NUM_FULLY_CONNECTED;
    output_layer.activation_function = TANH;
    output_layer.n_out = {OUTPUT_NEURONS, 1, 1};

    layer_data convolutional_layer;
    convolutional_layer.type = LAYER_NUM_CONVOLUTIONAL;
    convolutional_layer.stride_length = 1;
    convolutional_layer.receptive_field_length = 2;
    convolutional_layer.activation_function = LEAKY_RELU;
    convolutional_layer.n_out = {-1,-1, 3};

    layer_data fully_connected_layer_1;
    fully_connected_layer_1.type = LAYER_NUM_FULLY_CONNECTED;
    fully_connected_layer_1.activation_function = LEAKY_RELU;
    fully_connected_layer_1.n_out = {50, 1, 1};

    layer_data fully_connected_layer_2;
    fully_connected_layer_2.type = LAYER_NUM_FULLY_CONNECTED;
    fully_connected_layer_2.activation_function = LEAKY_RELU;
    fully_connected_layer_2.n_out = {50, 1, 1};
    // design the network
    int L = 5;
    layer_data* layers = new layer_data[L];
    layers[0] = input_layer;
    layers[L-1] = output_layer;
    layers[1] = convolutional_layer;
    layers[2] = fully_connected_layer_1;
    layers[3] = fully_connected_layer_2;

    // get hyperparams
    hyperparams params;

    // init DQN stuff
    player.batch_size = 8;
    int train_games = 1024;
    player.c = train_games/8;
    player.replay_buffer_size = 3000000;
    player.discount_factor = 0.95;
    player.epsilon_red = 0.99;
    player.epsilon = 1.0;

    convert_argv(argc, argv, player, train_games, params);

    // init networks
    player.main.init(layers, L, params);
    player.target.init(layers, L, params);

    for (int l = 0; l < L; l++) {
        cudaMemcpy(player.target.layers[l]->dev_weights, player.main.layers[l]->dev_weights, player.main.layers[l]->weights_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(player.target.layers[l]->dev_biases, player.main.layers[l]->dev_biases, player.main.layers[l]->biases_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    player.replay_buffer_counter = 0;
    player.replay_buffer.resize(player.replay_buffer_size);

    // train
    player.train(train_games);

    // save network
    std::string filename; std::cin >> filename;
    player.save("data/DQN/"+filename);
    std::cerr << "successfully saved DQN at " << filename << "\n";

    // free up memory
    player.main.clear();
    player.target.clear();
    delete[] layers;

    cudaDeviceReset();
    std::cout << "CUDA error: " << cudaGetErrorString(cudaPeekAtLastError()) << "\n";
}
