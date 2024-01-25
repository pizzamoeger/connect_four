#include "../game.h"

float MCTS::UCB1(int128 v, int128 p) {
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
        select(board);

        if (board.win() || board.turns == INPUT_NEURONS_W*INPUT_NEURONS_H) {
            int result = board.win() * simulations; // win every time
            backup(board.game_state, result);
            continue;
        }
        expand(board);
        int result = 0;

        Player* player;
        if (random_simulation) player = new Random();
        else player = new Almost_random();

        for (int j = 0; j < simulations; j++) {
            int res = simulate(board, player);
            if (res == board.turn) result++;
            else if (res == -board.turn) result--;
            // else tie
        }

        delete player;
        backup(board.game_state, result);
    }
}

void MCTS::select(connect_four_board &board) {

    while (true) { // loop until node with unvisited children is reached
        int best_col = -1;
        int best_uct = 0;

        for (int i = 0; i < 7; i++) {

            if (sims.find(7*board.game_state+i+1) == sims.end() && board.board[0][i] == 0) { // node has unvisited children
                return;
            }

            int uct = UCB1(7 * board.game_state + i + 1, board.game_state); // find best UCB value of children
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

        if (board.win() || board.turns == INPUT_NEURONS_W*INPUT_NEURONS_H) { // game ended
            return;
        }
    }
}

void MCTS::expand(connect_four_board &board) {

    std::vector<int> unexplored_children; // find all unvisited children
    for (int i = 0; i < 7; i++) {
        if (sims.find(7*board.game_state+i+1) == sims.end() && board.board[0][i] == 0) {
            unexplored_children.push_back(i);
        }
    }

    int child = unexplored_children[rand()%unexplored_children.size()]; // select one at random
    board.selected_col = child;
    board.play();
}

int MCTS::simulate(connect_four_board board, Player* player) {

    while (!board.win() && board.turns < INPUT_NEURONS_W*INPUT_NEURONS_H) { // simulate until game ends
        board.selected_col = player->get_col(board, false);
        board.play();
    }

    if (board.win()) {
        return board.turn;
    }
    return 0;
}

void MCTS::backup(int128 game_state, int result) {
    while (game_state != 0) {
        if (sims.find(game_state) == sims.end()) {
            sims[game_state] = 0;
            wins[game_state] = 0;
        }
        sims[game_state] += simulations;
        wins[game_state] += result;
        game_state = get_parent(game_state);
        result *= -1; // switch result
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
        if ((val > best_val || best_cols.empty()) && board.board[0][i] == 0) {
            best_val = val;
            best_cols.clear();
            best_cols.push_back(i);
        } else if (val == best_val && board.board[0][i] == 0) { // same best found
            best_cols.push_back(i);
        }
    }

    int best_col = best_cols[rand()%best_cols.size()];
    return best_col;
}

void MCTS::save(std::string filename) {
    if (!training) { // reset
        sims = init_sims;
        wins = init_wins;
    }

    if (sims.empty()) {
        sims[0] = 1;
        wins[0] = 0;
    }

    std::ofstream out(filename);
    out << sims.size() << "\n";
    for (auto [k, v] : sims) {
        out << k << " " << v << " ";
    }
    out << "\n";
    for (auto [k, v] : wins) {
        out << k << " " << v << " ";
    }
    out << "\n";

    out << random_simulation << " " << simulations << " " << iterations << " " << c << "\n";
    out << elo << "\n";
    out.close();
}

void MCTS::load(std::string filename) {
    std::ifstream in(filename);

    int size; in >> size;
    for (int i = 0; i < 2; i++) {
        for (int elem = 0; elem < size; elem++) {
            int128 key = 0;
            int val = 0;
            in >> key;
            in >> val;

            if (i == 0) init_sims[key] = val;
            else init_wins[key] = val;
        }
    }

    in >> random_simulation >> simulations >> iterations >> c;
    in >> elo;

    wins = init_wins;
    sims = init_sims;
}

void MCTS::train(int num_games) {
    for (int game = 0; game < num_games; game++) {
        if (game % 20 == 0) std::cerr << "Training in game " << game << "...\n";

        connect_four_board board;

        // play game
        while (true) {
            int col = get_col(board, false);
            board.selected_col = col;
            board.play();

            if (board.win() || board.turns == INPUT_NEURONS_W*INPUT_NEURONS_H) break;
        }
    }
}

int MCTS::get_col(connect_four_board board, bool eval) {
    run(board);
    return get_best_move(board);
}