#include "game.h"

void DQN::load(std::string filename) {
    return;
}

void DQN::save(std::string filename) {
    return;
}

int DQN::get_col(connect_four_board board) {
    int col = rand() % 7;
    while (board.board[0][col] != 0) col = rand() % 7;
    return col;
}