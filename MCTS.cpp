#include "game.h"

float MCTS::UCT(int v, int p) {
    if (sims.find(v) == sims.end() || sims.find(p) == sims.end() || wins.find(v) == wins.end()) return 0; // idk if this is the correct fix
    return wins[v]/sims[v] + c*sqrt(log(sims[p])/sims[v]);
}

int MCTS::get_parent(int v) {
    return (v-1)/7;
}

void MCTS::run(int num_roll_outs, connect_four_board board) {
    select(board); // select most promising leaf node from this state
    if (board.win() || board.turns == 42) return; // game ended (no need to expand)
    expand(board); // unexplored child of leaf node
    int result = 0;
    for (int i = 0; i < num_roll_outs; i++) {
        result += roll_out(board); // simulate
    }
    backup(board.game_state, result);
}

void MCTS::select(connect_four_board &board) {

    while (true) {
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
        play(board);

        if (board.win() || board.turns == 42) { // by always taking best UCT value, game will end
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
    int result = 0;

    while (!board.win() && board.turns < 42) { // simulate until game ends
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
        result = board.turn;
    }
    return result;
}

void MCTS::backup(int game_state, int result) {
    while (game_state != 0) {
        if (sims.find(game_state) == sims.end()) {
            sims[game_state] = 0;
            wins[game_state] = 0;
        }
        sims[game_state]++;
        wins[game_state] += result;
        game_state = get_parent(game_state);
    }
    if (sims.find(0) == sims.end()) {
        sims[0] = 0;
        wins[0] = 0;
    }
    sims[0]++;
    wins[0] += result;
}

int MCTS::get_best_move(connect_four_board board) {
    int best_col = -1;
    int best_uct = 0;

    for (int i = 0; i < 7; i++) {

        int uct = UCT(7*board.game_state+i+1, board.game_state); // find best UCT value of children
        if ((uct > best_uct || best_col == -1) && board.board[0][i] == 0) {
            best_uct = uct;
            best_col = i;
        }

    }

    return best_col;
}

void MCTS::play(connect_four_board &board) {
    while (board.board[board.selected_row][board.selected_col] != 0) {
        board.selected_row--;
        if (board.selected_row < 0) break;
    }
    if (board.selected_row >= 0) {
        board.turns++;
        board.turn = -board.turn;
        board.selected_row = 5;
        board.board[board.selected_row][board.selected_col] = board.turn;
    }
}

void MCTS::save() {
    std::string filename = "mcts.txt";
    std::ofstream out(filename);
    for (auto [k, v] : sims) out << k << ":" << v << " ";
    out << "\n";
    for (auto [k, v] : wins) out << k << ":" << v << " ";
    out.close();
}

void MCTS::load() {
    std::string filename = "mcts.txt";
    std::ifstream in(filename);
    // in file there are two lines, one for sims and one for wins
    // each line is a list of key:value pairs separated by spaces
    for (int i = 0; i < 2; i++) {
        std::string line;
        std::getline(in, line);
        int key = 0;
        int cur = 0;
        int neg = 1;
        for (char c : line) {
            if (c == ' ') {
                if (i == 0) sims[key] = cur*neg;
                else wins[key] = cur*neg;
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