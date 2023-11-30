#include "../game.h"

void DQL::load(std::string filename) {
    params = get_params();
    params.training_data_size = batch_size;
    params.test_data_size = 0;
    replay_buffer_size = 10;
    replay_buffer.resize(replay_buffer_size);
    epsilon = 0.5;
    discount_factor = 0.8;

    layer_data* layers;
    main.load(filename, params, layers);

    // copy everything from main to target
    target.init(layers, main.L, params);
    for (int l = 0; l < main.L; l++) {
        cudaMemcpy(target.layers[l]->dev_weights, main.layers[l]->dev_weights, main.layers[l]->weights_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(target.layers[l]->dev_biases, main.layers[l]->dev_biases, main.layers[l]->biases_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // TODO read last line -> elo

    delete[] layers;
    return;
}

int DQL::epsilon_greedy(float *out) {
    float r = (float)(rand()) / (float)(RAND_MAX);
    if (r < epsilon) {
       return rand() % OUTPUT_NEURONS;
    } else {
        int max_index = 0;
        for (int neuron = 0; neuron < OUTPUT_NEURONS; neuron++) {
            if (out[neuron] > out[max_index]) max_index = neuron;
        }
        return max_index;
    }
}

void DQL::save(std::string filename) {
    main.save(filename);

    // TODO save elo
    return;
}

float* DQL::feedforward(connect_four_board board, Network& net) {
    float* in = new float[INPUT_NEURONS];
    for (int row = 0; row < INPUT_NEURONS_X; row++) {
        for (int col = 0; col < INPUT_NEURONS_Y; col++) {
            in[row*INPUT_NEURONS_Y + col] = board.board[row][col];
        }
    }

    // feedforward
    net.feedforward(in, net.activations, net.derivatives_z);

    // get output
    float* out = new float[OUTPUT_NEURONS];
    cudaMemcpy(out, &net.activations[net.layers[net.L-1]->data.elems], 7*sizeof(float), cudaMemcpyDeviceToHost);

    delete[] in;
    return out;
}

int DQL::get_col(connect_four_board board) {
    float* out = feedforward(board, main);

    // select best action using epsilon greedy
    int action = epsilon_greedy(out);

    // get new state
    connect_four_board new_board = board;
    new_board.selected_col = action;
    new_board.play();

    // get reward in next state
    float reward = 0.0;
    if (new_board.win()) reward = 1.0;
    if (new_board.get_row() < 0) reward = -10.0; // invalid move

    // store in replay buffer
    int index = replay_buffer_counter%replay_buffer_size;
    replay_buffer[index] = {board, action, reward, new_board};

    // perform SGD on sample batch from replay buffer
    if (replay_buffer_counter >= replay_buffer_size) {
        std::vector<std::pair<float*, float*>> test_data = {};
        std::vector<std::pair<float*, float*>> training_data = {};

        std::vector<int> indices (replay_buffer_size);
        std::iota(indices.begin(), indices.end(), 0);
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));

        for (int experience = 0; experience < batch_size; experience++) {

            connect_four_board state = std::get<0>(replay_buffer[indices[experience]]);
            int action = std::get<1>(replay_buffer[indices[experience]]);
            float reward = std::get<2>(replay_buffer[indices[experience]]);
            connect_four_board new_state = std::get<3>(replay_buffer[indices[experience]]);

            // loss should be 0 everywhere but at index of action, there we expect it to be reward - max possible in next state
            float* replay_out = feedforward(state, main);
            float* tar_out = feedforward(new_state, target);

            float max_out = tar_out[0];
            for (int neuron = 0; neuron < OUTPUT_NEURONS; neuron++) max_out = std::max(max_out, tar_out[neuron]);
            replay_out[action] = reward - discount_factor*max_out;

            // get corresponding input
            float* replay_in = new float[INPUT_NEURONS];
            for (int row = 0; row < INPUT_NEURONS_X; row++) {
                for (int col = 0; col < INPUT_NEURONS_Y; col++) replay_in[row*INPUT_NEURONS_Y + col] = board.board[row][col];
            }

            training_data.push_back({replay_in, replay_out});
        }

        main.SGD(training_data, test_data);

        for (int i = 0; i < batch_size; i++) {
            delete[] training_data[i].first;
            delete[] training_data[i].second;
        }
    }

    delete[] out;
    return action;
}

void DQL::train(int num_games) {
   for (int game = 0; game < num_games; game++) {
       if (game % 20 == 0) std::cerr << "Game: " << game << "\n";

       // start state = new connect four board
       connect_four_board board;

       // play game
       while (true) {
           // get action
           int action = get_col(board);

           // execute action
           board.selected_col = action;
           board.play();

           if (game % 100 == 0) {
               for (int i = 0; i < 6; i++) {
                   for (int j = 0; j < 7; j++) {
                       if (board.board[i][j] == -1) std::cerr << "x";
                       else if (board.board[i][j] == 1) std::cerr << "o";
                       else std::cerr << "_";
                       std::cerr << " ";
                   }
                   std::cerr << "\n";
               }
               std::cerr << "\n\n\n";
           }

           // copy weights and biases every c steps
           if (game % c == 0) {
               for (int l = 0; l < main.L; l++) {
                   cudaMemcpy(target.layers[l]->dev_weights, main.layers[l]->dev_weights, main.layers[l]->weights_size * sizeof(float), cudaMemcpyDeviceToDevice);
                   cudaMemcpy(target.layers[l]->dev_biases, main.layers[l]->dev_biases, main.layers[l]->biases_size * sizeof(float), cudaMemcpyDeviceToDevice);
               }
           }

           replay_buffer_counter++;
           if (board.win() || board.turns == 42) break;
       }

   }

}