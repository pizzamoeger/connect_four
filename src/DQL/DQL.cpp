#include "../game.h"

void DQL::load(std::string filename) {
    return;
}

void DQL::save(std::string filename) {
    return;
}

int DQL::get_col(connect_four_board board) {
    int col = rand() % 7;
    while (board.board[0][col] != 0) col = rand() % 7;
    return col;
}