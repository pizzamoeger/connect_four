#include "../game.h"
#include <boost/multiprecision/cpp_int.hpp>

float MCTS::UCT(int128 v, int128 p) {
    if (sims.find(v) == sims.end() || sims.find(p) == sims.end() || wins.find(v) == wins.end()) return 0;
    return wins[v]/sims[v] + c*sqrt(log(sims[p])/sims[v]);
}

int128 MCTS::get_parent(int128 v) {
    return (v-1)/7;
}

void MCTS::run(connect_four_board board) {
    connect_four_board old_board = board;
    for (int i = 0; i < iterations; i++) {
        board = old_board;
        select(board); // select most promising leaf node from this state
        if (board.win() || board.turns == 42) {
            int result = board.win()*num_roll_outs; // win every time
            backup(board.game_state, result); // backup result
            continue; // game ended (no need to expand)
        }

        expand(board); // unexplored child of leaf node
        int result = 0;

        Player* player;
        if (random_roll_out) player = new Random(); // random rollout
        else player = new Almost_random(); // random rollout with some logic

        for (int j = 0; j < num_roll_outs; j++) { // simulate
            int r_out = roll_out(board, player); // get result
            if (r_out == board.turn) result++; // loss
            else if (r_out == -board.turn) result--; // win
            // else tie
        }

        delete player;
        backup(board.game_state, result);
    }
}

void MCTS::select(connect_four_board &board) {

    while (true) { // loop until leaf node reached
        int best_col = -1;
        int best_uct = 0;

        for (int i = 0; i < 7; i++) {

            if (sims.find(7*board.game_state+i+1) == sims.end() && board.board[0][i] == 0) { // leaf node reached
                return;
            }

            int uct = UCT(7*board.game_state+i+1, board.game_state); // find best UCT value of children
            if ((uct > best_uct || best_col == -1) && board.board[0][i] == 0) {
                best_uct = uct;
                best_col = i;
            }

        }

        if (best_col == -1) {
            return;
        }

        board.selected_col = best_col;
        board.play(); // play selected move

        if (board.win() || board.turns == 42) { // game ended
            return;
        }
    }
}

void MCTS::expand(connect_four_board &board) {

    std::vector<int> unexplored_children; // find all unexplored children
    for (int i = 0; i < 7; i++) {
        if (sims.find(7*board.game_state+i+1) == sims.end() && board.board[0][i] == 0) {
            unexplored_children.push_back(i);
        }
    }

    int child = unexplored_children[rand()%unexplored_children.size()]; // select one at random
    board.selected_col = child;
    board.play();
}

int MCTS::roll_out(connect_four_board board, Player* player) {

    while (!board.win() && board.turns < 42) { // simulate until game ends
        board.selected_col = player->get_col(board);
        board.play();
    }

    if (board.win()) {
        return board.turn;
    }
    return 0;
}

void MCTS::backup(int128 game_state, float result) {
    while (game_state != 0) {
        if (sims.find(game_state) == sims.end()) {
            sims[game_state] = 0;
            wins[game_state] = 0;
        }
        sims[game_state]++;
        wins[game_state] += result;
        game_state = get_parent(game_state);
        result = -discount_factor*result; // switch result
    }
    if (sims.find(0) == sims.end()) {
        sims[0] = 0;
        wins[0] = 0;
    }
    sims[0]++;
    wins[0] += result;
}

int MCTS::get_best_move(connect_four_board board) {
    std::vector<int> best_cols;
    float best_val = 0;

    for (int i = 0; i < 7; i++) {

        float val = 0; // find most promising child
        if (sims.find(7*board.game_state+i+1) != sims.end()) val = float(wins[7*board.game_state+i+1])/sims[7*board.game_state+i+1];
        if ((val > best_val || best_cols.empty()) && board.board[0][i] == 0) { // new best found
            best_val = val;
            best_cols.clear();
            best_cols.push_back(i);
        } else if (val == best_val && board.board[0][i] == 0) { // same best found
            best_cols.push_back(i);
        }

    }

    int best_col = best_cols[rand()%best_cols.size()]; // select one at random
    return best_col;
}

void MCTS::save(std::string filename) {
    if (!training) { // reset
        sims = init_sims;
        wins = init_wins;
    }

    std::ofstream out(filename);
    for (auto [k, v] : sims) {
        out << k << ":" << v << " ";
    }
    out << "\n";
    for (auto [k, v] : wins) {
        out << k << ":" << int(v) << " ";
    }
    out << "\n";

    out << random_roll_out << " " << num_roll_outs << " " << iterations << " " << c << " " << discount_factor << "\n";
    out << elo << "\n";
    out.close();
}

void MCTS::load(std::string filename) {
    std::ifstream in(filename);

    for (int i = 0; i < 2; i++) {
        std::string line;
        std::getline(in, line);
        int128 key = 0;
        int128 cur = 0;
        int neg = 1;
        for (char c : line) {
            if (c == ' ') {
                cur = cur*neg;
                if (i == 0) init_sims[key] = int(cur);
                else init_wins[key] = int(cur);
                cur = 0;
                neg = 1;
                continue;
            }
            if (c == ':') {
                key = cur;
                cur = 0;
                neg = 1;
                continue;
            }
            if (c == '-') {
                neg = -1;
                continue;
            }
            cur *= 10;
            cur += c-'0';
        }
    }

    in >> random_roll_out >> num_roll_outs >> iterations >> c >> discount_factor;
    in >> elo;
    in.close();

    wins = init_wins;
    sims = init_sims;
}

void MCTS::train(int num_games) {
    for (int i = 0; i < num_games; i++) {
        if (i % 20 == 0) std::cerr << "Game: " << i << "\n";

        connect_four_board board;

        // play game
        while (true) {
            int col = get_col(board);
            board.selected_col = col;
            board.play();

            if (board.win() || board.turns == 42) break;
        }
    }
}

int MCTS::get_col(connect_four_board board) {
    run(board);
    return get_best_move(board);
}