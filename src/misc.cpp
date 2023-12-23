#include "game.h"

std::pair<float,float> update_elo(float elo_1, float elo_2, int result) {
    const int K = 32;
    double expected_1 = 1.0 / (1.0 + pow(10.0, (elo_2 - elo_1) / 400.0));
    double expected_2 = 1.0 / (1.0 + pow(10.0, (elo_1 - elo_2) / 400.0));

    // TODO: could be written nicer
    float score_1;
    float score_2;
    if (result == 1) {
        score_1 = 0;
        score_2 = 1;
    } else if (result == -1) {
        score_1 = 1;
        score_2 = 0;
    } else {
        score_2 = 0.5;
        score_1 = 0.5;
    }

    int new_elo_1 = round(elo_1 + K * (score_1 - expected_1));
    int new_elo_2 = round(elo_2 + K * (score_2 - expected_2));

    return {new_elo_1, new_elo_2};
}

bool connect_four_board::win() {
    // place of last tile
    int placed_row = get_row()+1;
    int placed_col = selected_col;
    int turn = board[placed_row][placed_col];

    // length of continuous pattern of correct tile in dist xstp ystp
    auto check = [&](int xstp, int ystp) {
        int col = placed_col, row = placed_row;
        while (0 <= col && col < 7 && 0 <= row && row < 6 && board[row][col] == turn) {
            col += xstp;
            row += ystp;
        }
        return std::max(abs(col - placed_col), abs(row - placed_row)); // length
    };

    // look for all possible patterns
    for (int xstp: {0, 1}) { // no -1 needed as this will be covered by -xstp, -ystp
        for (int ystp: {-1, 0, 1}) {
            if (xstp == 0 && ystp == 0) continue;
            if (check(xstp, ystp) + check(-xstp, -ystp) >= /*5*/5) return true;
        }
    }
    return false;
}

int connect_four_board::get_row() { // return -1 if invalid
    int row = 5;
    while (board[row][selected_col] != 0) {
        row--;
        if (row < 0) break;
    }
    return row;
}

void connect_four_board::play() {
    selected_row = get_row();
    // play move
    if (selected_row >= 0) {
        board[selected_row][selected_col] = turn;
        game_state = 7*game_state+selected_col+1;
        turns++;
        turn = -turn;
        selected_row = 5;
    }
}

connect_four_board::connect_four_board() {
    // init board
    for (int i = 0; i < 6; i++) for (int j = 0; j < 7; j++) board[i][j] = 0;
    game_state = 0;
    turn = 1;
    turns = 0;
    selected_row = 5;
    selected_col = 0;
}