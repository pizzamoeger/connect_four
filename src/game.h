#ifndef CONNECT_FOUR_GAME_H
#define CONNECT_FOUR_GAME_H

#include "includes.h"
#include "neuralnetwork/Network.h"

#include <boost/multiprecision/cpp_int.hpp>
#define int128 boost::multiprecision::int128_t
#define EXIT_STR "EXIT!" // ! cannot be given as input
//#define int128 int

struct connect_four_board {
    int turn;
    int turns;
    int selected_col;
    int selected_row;
    int board[6][7];
    int128 game_state;

    connect_four_board();
    bool win();
    int get_row();
    void play();
};

struct Player {
    float elo = 1000.0;

    virtual ~Player() = default;
    virtual int get_col(connect_four_board board) = 0;
    virtual void load(std::string filename) = 0;
    virtual void save(std::string filename) = 0;
};

struct MCTS : public Player {
    std::map<int128, float> wins;
    std::map<int128, int> sims;

    std::map<int128, float> init_wins;
    std::map<int128, int> init_sims;
    bool training = false;

    bool random_roll_out = true;
    int num_roll_outs = 100;
    int iterations = 100;

    float c = sqrt(2.0f);
    float discount_factor = 1; // TODO: this is not functional yet

    int get_col(connect_four_board board);

    float UCT(int128 v, int128 p);
    int128 get_parent(int128 v);

    void run(connect_four_board board);
    void select(connect_four_board &board);
    void expand(connect_four_board &board);
    int roll_out(connect_four_board board);
    int roll_out_rand(connect_four_board board);
    void backup(int128 game_state, float result);

    int get_best_move(connect_four_board board);

    void save(std::string filename = "MCTS/bot.txt");
    void load(std::string filename = "MCTS/bot.txt");
    void train(int num_games);
    std::vector<int> can_win(int player, connect_four_board board);
};

struct DQL : public Player {
    Network main;
    Network target;
    hyperparams params;

    int c;
    float epsilon;
    int replay_buffer_size;
    int replay_buffer_counter;
    int batch_size;

    float discount_factor;

    std::vector<std::tuple<connect_four_board, int, float, connect_four_board>> replay_buffer;

    int epsilon_greedy(float* out);
    float* feedforward(connect_four_board board, Network& net);

    int get_col(connect_four_board board);
    void load(std::string filename = "DQL/bot.txt");
    void save(std::string filename = "DQL/bot.txt");

    void train(int num_games);
};

struct Random : public Player {
    int get_col(connect_four_board board);
    void save(std::string filename = "data/RANDOM/bot.txt");
    void load(std::string filename = "data/RANDOM/bot.txt");
};

struct Almost_random : public Player {
    int get_col(connect_four_board board);
    void save(std::string filename = "data/ALMOST_RANDOM/bot.txt");
    void load(std::string filename = "data/ALMOST_RANDOM/bot.txt");

    std::vector<int> can_win(int player, connect_four_board board);
};

struct Human : public Player {
    int get_col(connect_four_board board) {
        return 0;
    };
    void save(std::string filename = "HUMAN/test.txt");
    void load(std::string filename = "HUMAN/test.txt");
};

#endif //CONNECT_FOUR_GAME_H
