#include "../game.h"

void DQN::load(std::string filename) {
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
    std::string line;
    while (getline(file, line));
    elo = atoi(line.c_str());
    file.close();

    delete[] layers;
    return;
}

int DQN::epsilon_greedy(float *out, connect_four_board board, bool eval) {
    float r = (float)(rand()) / (float)(RAND_MAX);

    //assert(epsilon == 0 || !eval); // in eval, epsilon should be zero
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
            if (eval) {
                board.selected_col = neuron;
                if (board.get_row() < 0) continue;
            }
            if (max_index == -1 || out[neuron] > out[max_index]) {
                max_index = neuron;
                max_indices = {max_index};
            } else if (out[neuron] == out[max_index]) max_indices.push_back(neuron);
        }

        return max_indices[rand()%max_indices.size()];
    }
}

void DQN::save(std::string filename) {
    main.save(filename);

    // save elo
    std::ofstream file;
    file.open(filename, std::ios_base::app);
    file << elo;
    file.close();
    return;
}

float* DQN::get_input(connect_four_board board) {
    // copy board to activations
    float* in = new float[INPUT_NEURONS];
    for (int row = 0; row < INPUT_NEURONS_H; row++) {
        for (int col = 0; col < INPUT_NEURONS_W; col++) {
            in[row*INPUT_NEURONS_W + col] = (board.board[row][col]==board.turn);
            in[row*INPUT_NEURONS_W + col + INPUT_NEURONS_W * INPUT_NEURONS_H] = (board.board[row][col]==(-board.turn));
        }
    }

    // copy to device
    float* dev_in;
    cudaMalloc(&dev_in, INPUT_NEURONS*sizeof(float));
    cudaMemcpy(dev_in, in, INPUT_NEURONS*sizeof(float), cudaMemcpyHostToDevice);

    delete[] in;
    return dev_in;
}

float* DQN::get_output(Dev_experience exp) {
    main.feedforward(exp.state, main.activations, main.derivatives_z);
    target.feedforward(exp.new_state, target.activations, target.derivatives_z);

    float* dev_out;
    cudaMalloc(&dev_out, OUTPUT_NEURONS*sizeof(float));
    get_dqn_out<<<OUTPUT_NEURONS,1>>> (&main.activations[main.layers[main.L-1]->data.elems], &target.activations[target.layers[target.L-1]->data.elems], exp.reward, exp.action, dev_out, dev_discount_factor);
    return dev_out;
}

float* DQN::feedforward(connect_four_board board, Network& net) {
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

std::vector<Dev_experience> DQN::get_random_batch() {
    // get the indices of the possible experiences and shuffle them
    std::vector<int> indices (std::min(replay_buffer_size,replay_buffer_counter));
    std::iota(indices.begin(), indices.end(), 0);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));

    std::vector<Dev_experience> batch (batch_size);
    for (int index = 0; index < batch_size; index++) {
        batch[index] = replay_buffer[indices[index]];
    }
    return batch;
}

Dev_experience DQN::get_experience(connect_four_board board, int action) {
    // get new state
    connect_four_board new_board = board;
    new_board.selected_col = action;
    new_board.play();

    float* state = get_input(board);
    float* new_state = get_input(new_board);

    float reward = 0.0;
    if (new_board.get_row() < 0) reward = -1.0; // illegal move
    else if (new_board.win()) reward = 1.0; // winning move

    int* dev_action;
    cudaMalloc(&dev_action, sizeof(int));
    cudaMemcpy(dev_action, &action, sizeof(int), cudaMemcpyHostToDevice);

    float* dev_reward;
    cudaMalloc(&dev_reward, sizeof(float));
    cudaMemcpy(dev_reward, &reward, sizeof(float), cudaMemcpyHostToDevice);

    return {state, dev_action, dev_reward, new_state};
}


void DQN::store_in_replay_buffer(Dev_experience exp) {
    int index = replay_buffer_counter % replay_buffer_size;

    cudaFree(replay_buffer[index].state);
    cudaFree(replay_buffer[index].action);
    cudaFree(replay_buffer[index].reward);
    cudaFree(replay_buffer[index].new_state);

    replay_buffer[index] = exp;
    replay_buffer_counter++;
}

int DQN::get_col(connect_four_board board, bool eval) {
    float* out = feedforward(board, main);

    // select best action using epsilon greedy
    int action = epsilon_greedy(out, board, eval);

    delete[] out;
    return action;
}

void DQN::copy_main_to_target() {
    for (int l = 0; l < main.L; l++) {
        cudaMemcpy(target.layers[l]->dev_weights, main.layers[l]->dev_weights,
                   main.layers[l]->weights_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(target.layers[l]->dev_biases, main.layers[l]->dev_biases,
                   main.layers[l]->biases_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

void DQN::train(int num_games) {
    int turn = 0;
    for (int game = 0; game < num_games; game++) {
       connect_four_board board;

        if (game%20 == 0) std::cerr << "Game " << game << "\n";

       // play game
       while (true) {
           int action = get_col(board, false);
           Dev_experience exp = get_experience(board, action);
           store_in_replay_buffer(exp);

           if (replay_buffer_counter >= batch_size) {
               std::vector<Dev_experience> batch = get_random_batch();

               for (auto experience: batch) {
                   // train experience from batch
                   float *dev_out = get_output(experience);

                   main.SGD({{experience.state, dev_out}}, {});

                   cudaFree(dev_out);
               }
           }

           // execute action
           board.selected_col = action;
           board.play();

           // copy weights and biases every c steps
           if (turn % c == 0) copy_main_to_target();
           turn++;

           if (board.win() || board.turns == INPUT_NEURONS_W*INPUT_NEURONS_H) break;
       }
       epsilon *= epsilon_red;
       epsilon = std::max(epsilon, 0.2f);
   }
}
