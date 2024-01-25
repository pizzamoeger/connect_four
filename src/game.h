#ifndef CONNECT_FOUR_GAME_H
#define CONNECT_FOUR_GAME_H

#include "includes.h"
#include "neuralnetwork/Network.h"

#define EXIT_STR "EXIT!" // ! cannot be given as input
#define int128 __int128

struct connect_four_board {
    int turn;
    int turns;
    int selected_col;
    int selected_row;
    int board[INPUT_NEURONS_H][INPUT_NEURONS_W];
    int128 game_state;

    connect_four_board();
    bool win();
    int get_row();
    void play();

    friend std::ostream& operator<<(std::ostream& os, const connect_four_board& board);
};

struct Dev_experience {
    float* state;
    int* action;
    float* reward;
    float* new_state;
};

struct Player {
    float elo = 1000.0;

    // eval specifies whether player is currently being evaluated or trained
    virtual int get_col(connect_four_board board, bool eval) = 0;
    virtual void load(std::string filename) = 0;
    virtual void save(std::string filename) = 0;
    virtual void train(int num_games) {}
};

struct MCTS : public Player {
    std::map<int128, int> wins;
    std::map<int128, int> sims;

    std::map<int128, int> init_wins;
    std::map<int128, int> init_sims;
    bool training = false;

    bool random_simulation = true;
    int simulations = 100;
    int iterations = 100;
    float c = sqrt(2.0f);

    float UCB1(int128 v, int128 p);
    int128 get_parent(int128 v);

    void run(connect_four_board board);
    void select(connect_four_board &board);
    void expand(connect_four_board &board);
    int simulate(connect_four_board board, Player* player);
    void backup(int128 game_state, int result);

    int get_best_move(connect_four_board board);

    int get_col(connect_four_board board, bool eval);
    void save(std::string filename = "MCTS/bot.txt");
    void load(std::string filename = "MCTS/bot.txt");
    void train(int num_games);
};

struct DQN : public Player {
    Network main;
    Network target;
    hyperparams params;

    int c = 8;
    int batch_size = 8;
    float epsilon = 0;
    float epsilon_red = 0.99;
    int replay_buffer_size = 16;
    int replay_buffer_counter = 0;
    float discount_factor = 0.95;
    float* dev_discount_factor;

    std::vector<Dev_experience> replay_buffer;

    int epsilon_greedy(float* out, connect_four_board board, bool eval);
    float* get_input(connect_four_board board);
    float* get_output(Dev_experience exp);
    std::vector<Dev_experience> get_random_batch();
    Dev_experience get_experience(connect_four_board board, int action);
    void store_in_replay_buffer(Dev_experience exp);
    void copy_main_to_target();
    float* feedforward(connect_four_board board, Network& net);

    int get_col(connect_four_board board, bool eval);
    void load(std::string filename = "DQN/bot.txt");
    void save(std::string filename = "DQN/bot.txt");

    void train(int num_games);
};

struct Random : public Player {
    int get_col(connect_four_board board, bool eval);
    void save(std::string filename = "data/RANDOM/bot.txt");
    void load(std::string filename = "data/RANDOM/bot.txt");
};

struct Almost_random : public Player {
    int get_col(connect_four_board board, bool eval);
    void save(std::string filename = "data/ALMOST_RANDOM/bot.txt");
    void load(std::string filename = "data/ALMOST_RANDOM/bot.txt");

    std::vector<int> can_win(int player, connect_four_board board);
};

struct Human : public Player {
    int get_col(connect_four_board board, bool eval) {return 0;}
    void save(std::string filename = "HUMAN/test.txt");
    void load(std::string filename = "HUMAN/test.txt");
};

std::pair<float,float> update_elo(float elo_1, float elo_2, int result);

std::istream& operator>>(std::istream& in, int128& num);
std::ostream& operator<<(std::ostream& os, const int128& num);

#endif //CONNECT_FOUR_GAME_H
