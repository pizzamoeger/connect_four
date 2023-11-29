#include "../game.h"

void DQL::load(std::string filename) {
    // if filename is "NEW" or something like this, no load, just init
    main.load(filename, params);

    // copy everything from main to target

    // read last line -> elo

    // load replay buffer?
    return;
}

void DQL::save(std::string filename) {
    // store replay buffer?
    main.save(filename);
    return;
}

int DQL::get_col(connect_four_board board) {
    // feedforward
    // select best action using epsilon greedy
    // perform SGL on sample batch from replay buffer, store in replay buffer
    // return the selected action
    // here, target will never get adjusted. (or all the time depending how often it is stored and loaded again.)
    int col = rand() % 7;
    while (board.board[0][col] != 0) col = rand() % 7;
    return col;
}

void DQL::train(int num_games) {
   for (int game = 0; game < num_games; game++) {
        // start state = new connect four board
        // while game is not over
            // feedforward in main, select best action with epsilon greedy, store in replay buffer
            // SGD on mainwith replay buffer
       if (game % c == 0) {}// copy weights and biases from main to target
   }

}