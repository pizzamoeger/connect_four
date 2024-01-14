#include "../game.h"

void DQL::load(std::string filename) {
    params = get_params();
    params.training_data_size = batch_size;
    params.test_data_size = 0;
    replay_buffer_size = 10;
    replay_buffer.resize(replay_buffer_size);
    epsilon = 0;
    discount_factor = 0.8;

    layer_data* layers;
    main.load(filename, params, layers);

    // copy everything from main to target
    target.init(layers, main.L, params);
    for (int l = 0; l < main.L; l++) {
        cudaMemcpy(target.layers[l]->dev_weights, main.layers[l]->dev_weights, main.layers[l]->weights_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(target.layers[l]->dev_biases, main.layers[l]->dev_biases, main.layers[l]->biases_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // get ELO from last line
    std::ifstream file(filename);
    while (file >> elo);
    file.close();

    delete[] layers;
    return;
}

int DQL::epsilon_greedy(float *out, connect_four_board board) {
    float r = (float)(rand()) / (float)(RAND_MAX);

    if (r < epsilon) {
        std::vector<int> possible_moves;
        for (int i = 0; i < OUTPUT_NEURONS; i++) {
            board.selected_col = i;
            if (board.get_row() < 0) continue;
            possible_moves.push_back(i);
        }
       return possible_moves[rand()%possible_moves.size()];
    } else {
        std::vector<int> max_indices;
        int max_index = -1;
        for (int neuron = 0; neuron < OUTPUT_NEURONS; neuron++) {
            board.selected_col = neuron;
            if (board.get_row() < 0) continue;
            if (max_index == -1 || out[neuron] > out[max_index]) {
                max_index = neuron;
                max_indices = {max_index};
            } else if (out[neuron] == out[max_index]) max_indices.push_back(neuron);
        }

        return max_indices[rand()%max_indices.size()];
    }
}

void DQL::save(std::string filename) {
    main.save(filename);

    // save elo
    std::ofstream file;
    file.open(filename, std::ios_base::app); // append instead of overwrite
    file << elo;
    file.close();
    return;
}

float* DQL::get_input(connect_four_board board) {
    // copy board to activations
    float* in = new float[INPUT_NEURONS];
    for (int row = 0; row < INPUT_NEURONS_H; row++) {
        for (int col = 0; col < INPUT_NEURONS_W; col++) {
            in[row*INPUT_NEURONS_W + col] = board.board[row][col];
        }
    }

    std::cout << board;

    // copy to device
    float* dev_in;
    cudaMalloc(&dev_in, INPUT_NEURONS*sizeof(float));
    cudaMemcpy(dev_in, in, INPUT_NEURONS*sizeof(float), cudaMemcpyHostToDevice);

    delete[] in;
    return dev_in;
}

float* DQL::get_output(Experience exp) {
    float* main_out = feedforward(exp.state, main);
    float* tar_out = feedforward(exp.new_state, target);

    float max_out = tar_out[0];
    for (int neuron = 0; neuron < OUTPUT_NEURONS; neuron++) max_out = std::max(max_out, tar_out[neuron]);

    main_out[exp.action] = exp.reward - discount_factor*max_out;

    float* dev_out;
    cudaMalloc(&dev_out, OUTPUT_NEURONS*sizeof(float));
    cudaMemcpy(dev_out, main_out, OUTPUT_NEURONS*sizeof(float), cudaMemcpyHostToDevice);

    delete[] main_out;
    delete[] tar_out;
    return dev_out;
}

float* DQL::feedforward(connect_four_board board, Network& net) {
    // get input
    float* in = get_input(board);

    // feedforward
    net.feedforward(in, net.activations, net.derivatives_z);

    // get output
    float* out = new float[OUTPUT_NEURONS];
    cudaMemcpy(out, &net.activations[net.layers[net.L-1]->data.elems], OUTPUT_NEURONS*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(in);
    return out;
}

std::vector<Experience> DQL::get_random_batch() {
    // get the indices of the possible experiences and shuffle them
    std::vector<int> indices (std::min(replay_buffer_size,replay_buffer_counter));
    std::iota(indices.begin(), indices.end(), 0);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));

    std::vector<Experience> batch (batch_size);
    for (int index = 0; index < batch_size; index++) {
        batch[index] = replay_buffer[indices[index]];
    }
    return batch;
}

Experience DQL::get_experience(connect_four_board board, int action) {
    // get new state
    connect_four_board new_board = board;
    new_board.selected_col = action;
    new_board.play();

    // get reward in next state
    float reward = 0.0;
    if (new_board.win()) reward = 1.0;

    return {board, action, reward, new_board};
}


void DQL::store_in_replay_buffer(Experience exp) {
    int index = replay_buffer_counter % replay_buffer_size;
    replay_buffer[index] = exp;
    replay_buffer_counter++;
}

int DQL::get_col(connect_four_board board) {
    for (int l = 0; l < main.L; l++) {
        float* weights = new float [main.layers[l]->weights_size];
        cudaMemcpy(weights, main.layers[l]->dev_weights, main.layers[l]->weights_size*sizeof(float), cudaMemcpyDeviceToHost);

        std::cerr << "layer: " << l << ": ";
        std::cerr << "dev weights: " << main.layers[l]->dev_weights << "\n";
        for (int w = 0; w < main.layers[l]->weights_size; w++) std::cerr << weights[w] << " ";
        std::cerr << "\n";

        delete[] weights;
    }

    float* out = feedforward(board, main);

    // select best action using epsilon greedy
    int action = epsilon_greedy(out, board);

    delete[] out;
    return action;
}

void DQL::copy_main_to_target() {
    for (int l = 0; l < main.L; l++) {
        cudaMemcpy(target.layers[l]->dev_weights, main.layers[l]->dev_weights,
                   main.layers[l]->weights_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(target.layers[l]->dev_biases, main.layers[l]->dev_biases,
                   main.layers[l]->biases_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

void DQL::train(int num_games) {
    for (int game = 0; game < num_games; game++) {
       connect_four_board board;

       // play game
       while (true) {
           int action = get_col(board);
           Experience exp = get_experience(board, action);
           store_in_replay_buffer(exp);

           if (replay_buffer_counter >= batch_size) {

               std::vector<Experience> batch = get_random_batch();

               for (auto experience: batch) {
                   float *dev_in = get_input(experience.state);
                   float *dev_out = get_output(experience);

                   main.SGD({{dev_in, dev_out}}, {});

                   cudaFree(dev_in);
                   cudaFree(dev_out);
               }
           }

           // execute action
           board = exp.new_state;

           // copy weights and biases every c steps
           if (game % c == 0) copy_main_to_target();

           if (board.win() || board.turns == INPUT_NEURONS) break;
       }
       epsilon = 0.9999*epsilon;
       epsilon = std::max(epsilon, 0.1f);
   }
}
