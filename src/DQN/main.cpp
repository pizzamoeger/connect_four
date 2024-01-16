#include "../game.h"
#include "../neuralnetwork/Network.h"

int main(int argc, char** argv) {

    // randomness
    srand(time(NULL));

    DQN player;

    // design layers
    layer_data input;
    input.type = LAYER_NUM_INPUT;
    input.n_out = {INPUT_NEURONS_H, INPUT_NEURONS_W, 1};

    layer_data convolutional;
    convolutional.type = LAYER_NUM_CONVOLUTIONAL;
    convolutional.stride_length = 1;
    convolutional.receptive_field_length = 4;
    convolutional.activation_function = LEAKY_RELU;
    convolutional.n_out = {-1,-1, 3};

    layer_data fully_connected1;
    fully_connected1.type = LAYER_NUM_FULLY_CONNECTED;
    fully_connected1.activation_function = LEAKY_RELU;
    fully_connected1.n_out = {50, 1, 1};

    layer_data fully_connected2;
    fully_connected2.type = LAYER_NUM_FULLY_CONNECTED;
    fully_connected2.activation_function = LEAKY_RELU;
    fully_connected2.n_out = {30, 1, 1};

    layer_data outt;
    outt.type = LAYER_NUM_FULLY_CONNECTED;
    outt.activation_function = TANH;
    outt.n_out = {OUTPUT_NEURONS, 1, 1};

    // design the network
    int L = 4;
    layer_data* layers = new layer_data[L];
    layers[0] = input;
    layers[1] = convolutional;
    layers[1] = convolutional;
    layers[1] = fully_connected1;
    layers[2] = fully_connected2;
    layers[3] = outt;

    // get hyperparams
    hyperparams params = get_params();

    if (argc == 7) {
        // read hyperparams from commandline arguments
        params.fully_connected_weights_learning_rate = atof(argv[1]);
        params.fully_connected_biases_learning_rate = atof(argv[2]);
        params.convolutional_weights_learning_rate = atof(argv[3]);
        params.convolutional_biases_learning_rate = atof(argv[4]);
        params.L2_regularization_term = atof(argv[5]);
        params.momentum_coefficient = atof(argv[6]);
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
    int train_games = 1024;
    player.c = 16;

    player.replay_buffer_size = 3000000;
    player.discount_factor = 0.95;
    player.epsilon_red = 0.99;
    player.epsilon = 1.0;
    player.replay_buffer_counter = 0;
    player.replay_buffer.resize(player.replay_buffer_size);

    // train
    player.train(train_games);

    // save network
    // FIND-TAG-STORING
    //std::cerr << "Where should the network be stored? "; std::string filename; std::cin >> filename;
    std::string filename = "data/DQN/batch_size/fc_";
    filename += std::to_string(player.batch_size);
    filename += "_" + std::to_string(player.c);
    filename += "_" + std::to_string(train_games);
    filename += ".txt1";

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
