#include "game.h"

int Random::get_col(connect_four_board board, bool eval) {
    std::vector<int> moves (INPUT_NEURONS_W);
    std::iota(moves.begin(), moves.end(), 0);
    std::random_shuffle(moves.begin(), moves.end());

    int ind = 0;
    board.selected_col = moves[ind];
    while (ind < INPUT_NEURONS_W && board.board[0][board.selected_col] != 0) {
        board.selected_col = moves[ind++];
    }

    return board.selected_col;
}

void Random::save(std::string filename) {
    filename = "data/RANDOM/bot.txt";
    std::ofstream out(filename);
    out << elo << "\n";
    out.close();
}

void Random::load(std::string filename) {
    filename = "data/RANDOM/bot.txt";
    std::ifstream in(filename);
    in >> elo;
    in.close();
}

std::vector<int> Almost_random::can_win(int player, connect_four_board board) {
    std::vector<int> winners;

    board.turn = player;
    connect_four_board new_board = board;

    for (int col = 0; col < INPUT_NEURONS_W; col++) {
        new_board.selected_col = col;
        new_board.selected_row = new_board.get_row();
        if (new_board.selected_row < 0) continue; // illegal move

        new_board.play(); // play move
        if (new_board.win()) winners.push_back(col); // if winning move, append
        new_board = board;
    }
    return winners;
}

int Almost_random::get_col(connect_four_board board, bool eval) {
    if (!can_win(board.turn, board).empty()) // can win in one move
        board.selected_col = can_win(board.turn, board)[0];
    else if (!can_win(-board.turn, board).empty()) // would lose if not playing this move
        board.selected_col = can_win(-board.turn, board)[0];
    else {
        std::vector<int> no_loss_moves;
        std::vector<int> valid_moves;

        for (int col = 0; col < INPUT_NEURONS_W; col++) {
            connect_four_board new_board = board;
            new_board.selected_col = col;
            new_board.selected_row = new_board.get_row();
            if (new_board.selected_row < 0) continue; // invalid

            new_board.play(); // play the move
            valid_moves.push_back(col); // theoretically possible

            if (can_win(new_board.turn, new_board).empty()) no_loss_moves.push_back(col);
        }
        if (no_loss_moves.empty()) board.selected_col = valid_moves[rand() % valid_moves.size()]; // will lose no matter what
        else board.selected_col = no_loss_moves[rand() % no_loss_moves.size()];
    }
    return board.selected_col;
}

void Almost_random::save(std::string filename) {
    filename = "data/ALMOST_RANDOM/bot.txt";
    std::ofstream out(filename);
    out << elo << "\n";
    out.close();
}

void Almost_random::load(std::string filename) {
    filename = "data/ALMOST_RANDOM/bot.txt";
    std::ifstream in(filename);
    in >> elo;
    in.close();
}

void Human::save(std::string filename) {
    std::ofstream out(filename);
    out << elo << "\n";
    out.close();
}

void Human::load(std::string filename) {
    std::ifstream in(filename);
    in >> elo;
    in.close();
}