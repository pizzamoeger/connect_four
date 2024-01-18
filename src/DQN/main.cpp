#include "../game.h"
#include "../neuralnetwork/Network.h"

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
    convolutional_layer.receptive_field_length = 4;
    convolutional_layer.activation_function = LEAKY_RELU;
    convolutional_layer.n_out = {-1,-1, 3};

    layer_data fully_connected_layer_1;
    fully_connected_layer_1.type = LAYER_NUM_FULLY_CONNECTED;
    fully_connected_layer_1.activation_function = LEAKY_RELU;
    fully_connected_layer_1.n_out = {50, 1, 1};

    layer_data fully_connected_layer_2;
    fully_connected_layer_2.type = LAYER_NUM_FULLY_CONNECTED;
    fully_connected_layer_2.activation_function = LEAKY_RELU;
    fully_connected_layer_2.n_out = {30, 1, 1};

    // design the network
    int L = 4;
    layer_data* layers = new layer_data[L];
    layers[0] = input_layer;
    layers[L-1] = output_layer;
    layers[1] = fully_connected_layer_1;
    layers[2] = fully_connected_layer_2;

    // get hyperparams
    hyperparams params;
    std::cout << "Enter neural network hyperparams manually? [Y/N (default)] ";
    char c; std::cin >> c;
    if (c == 'Y') {
        std::cout << "fully connected weights eta: "; std::cin >> params.fully_connected_weights_learning_rate;
        std::cout << "fully connected biases eta: "; std::cin >> params.fully_connected_biases_learning_rate;
        std::cout << "convolutional weights eta: "; std::cin >> params.convolutional_weights_learning_rate;
        std::cout << "convolutional biases eta: "; std::cin >> params.convolutional_biases_learning_rate;
        std::cout << "fully connected weights eta reduction: "; std::cin >> params.fcWRed;
        std::cout << "fully connected biases eta reduction: "; std::cin >> params.fcBRed;
        std::cout << "convolutional weights eta reduction: "; std::cin >> params.convWRed;
        std::cout << "convolutional biases eta reduction: "; std::cin >> params.convBRed;
        std::cout << "L2 regularization: "; std::cin >> params.L2_regularization_term;
        std::cout << "momentum coefficient: "; std::cin >> params.momentum_coefficient;
    }

    // init networks
    player.main.init(layers, L, params);
    player.target.init(layers, L, params);

    for (int l = 0; l < L; l++) {
        cudaMemcpy(player.target.layers[l]->dev_weights, player.main.layers[l]->dev_weights, player.main.layers[l]->weights_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(player.target.layers[l]->dev_biases, player.main.layers[l]->dev_biases, player.main.layers[l]->biases_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // init DQN stuff
    player.batch_size = 1;
    player.c = 16;
    int train_games = 1024;
    player.replay_buffer_size = 3000000;
    player.discount_factor = 0.95;
    player.epsilon_red = 0.99;
    player.epsilon = 1.0;

    std::cout << "Enter DQN hyperparams manually? [Y/N (default)] ";
    std::cin >> c;
    if (c == 'Y') {
        std::cout << "Batch size: "; std::cin >> player.batch_size;
        std::cout << "replay buffer size: "; std::cin >> player.replay_buffer_size;
        std::cout << "c: "; std::cin >> player.c;
        std::cout << "trainings games: "; std::cin >> train_games;
        std::cout << "discount factor: "; std::cin >> player.discount_factor;
        std::cout << "epsilon: "; std::cin >> player.epsilon;
        std::cout << "epsilon reduction: "; std::cin >> player.epsilon_red;
    }

    player.replay_buffer_counter = 0;
    player.replay_buffer.resize(player.replay_buffer_size);

    // train
    player.train(train_games);

    // save network
    // FIND-TAG-STORING
    //std::cerr << "Where should the network be stored? "; std::string filename; std::cin >> filename;
    std::string filename = "data/DQN/architecture/fc_";
    filename += std::to_string(player.batch_size);
    filename += "_" + std::to_string(player.c);
    filename += "_" + std::to_string(train_games);
    filename += ".txt";

    player.save(filename);
    std::ofstream file;
    std::cerr << "saved DQN at " << filename << "\n";

    // free up memory
    player.main.clear();
    player.target.clear();
    delete[] layers;

    cudaDeviceReset();
    std::cout << cudaGetErrorString(cudaPeekAtLastError()) << "\n";
}
