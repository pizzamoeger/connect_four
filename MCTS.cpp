#include "game.h"

float MCTS::UCT(int128 v, int128 p) {
    if (sims.find(v) == sims.end() || sims.find(p) == sims.end() || wins.find(v) == wins.end()) return 0;
    return wins[v]/sims[v] + c*sqrt(log(sims[p])/sims[v]);
}

int128 MCTS::get_parent(int128 v) {
    return (v-1)/7;
}

void MCTS::run(connect_four_board board) {
    for (int i = 0; i < iterations; i++) {
        select(board); // select most promising leaf node from this state
        if (board.win() || board.turns == 42) return; // game ended (no need to expand)
        expand(board); // unexplored child of leaf node
        int result = 0;
        for (int j = 0; j < num_roll_outs; j++) { // simulate
            if (roll_out(board) == board.turn) result++; // win
            else if (roll_out(board) == -board.turn) result--; // loss
        }
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

        board.game_state = 7*board.game_state+best_col+1;
        board.selected_col = best_col;
        play(board); // play selected move

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
    play(board);
    board.game_state = 7*board.game_state+child+1;
}

int MCTS::roll_out(connect_four_board board) {

    while (!board.win() && board.turns < 42) { // simulate until game ends by selecting random moves
        std::vector<int> moves (7);
        std::iota(moves.begin(), moves.end(), 0);
        std::random_shuffle(moves.begin(), moves.end());
        int ind = 0;
        board.selected_col = moves[ind];
        while (ind < 7 && board.board[0][board.selected_col] != 0) {
            board.selected_col = moves[ind++];
        }
        if (ind >= 7) break;
        play(board);
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
        result = -discount_factor*result; // switch result MAYBE ADD DISCOUNT FACTOR?
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
    int best_val = 0;


    for (int i = 0; i < 7; i++) {

        int val = 0; // find most promising child
        if (sims.find(7*board.game_state+i+1) != sims.end()) val = wins[7*board.game_state+i+1]/sims[7*board.game_state+i+1];
        if ((val > best_val || best_cols.empty()) && board.board[0][i] == 0) {
            best_val = val;
            best_cols.clear();
            best_cols.push_back(i);
        } else if (val == best_val && board.board[0][i] == 0) {
            best_cols.push_back(i);
        }

    }

    int best_col = best_cols[rand()%best_cols.size()]; // select one at random
    return best_col;
}

void MCTS::play(connect_four_board &board) {
    // find lowest empty row in selected column
    while (board.board[board.selected_row][board.selected_col] != 0) {
        board.selected_row--;
        if (board.selected_row < 0) break;
    }
    // play move
    if (board.selected_row >= 0) {
        board.board[board.selected_row][board.selected_col] = board.turn;
        board.turns++;
        board.turn = -board.turn;
        board.selected_row = 5;
    }
}

void MCTS::save(std::string filename) {
    std::ofstream out(filename);
    for (auto [k, v] : sims) {
        out << k << ":" << v << " ";
    }
    out << "\n";
    for (auto [k, v] : wins) {
        out << k << ":" << int(v) << " ";
    }
    out.close();
    std::cerr << filename<<"\n";
}

void MCTS::load() {
    std::string filename = "mcts.txt";
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
                if (i == 0) sims[key] = int(cur);
                else wins[key] = int(cur);
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
    in.close();
}

void MCTS::train(int num_games) {
    for (int i = 0; i < num_games; i++) {
        if (i % 1000 == 0) std::cerr << i << "\n";

        connect_four_board board;
        // init board
        for (int i = 0; i < 6; i++) for (int j = 0; j < 7; j++) board.board[i][j] = 0;
        board.game_state = 0;
        board.turn = 1;
        board.turns = 0;
        board.selected_row = 5;

        // play game
        while (true) {
            run(board);
            int col = get_best_move(board);
            board.selected_col = col;
            play(board);

            if (board.win() || board.turns == 42) break;
        }
    }
}