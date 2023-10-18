#ifndef CONNECT_FOUR_GAME_H
#define CONNECT_FOUR_GAME_H

#include <SDL2/SDL.h>
#include <iostream>
#include <cmath>
#include <SDL2/SDL_ttf.h>
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <boost/multiprecision/cpp_int.hpp>

#define int128 boost::multiprecision::int128_t
//#define int128 int

//Screen dimension constants
const int TEXT_SIZE = 60;
const int TEXT_DIST = 100;
const int SCREEN_WIDTH = 1000;
const int SCREEN_HEIGHT = 700+TEXT_DIST+TEXT_SIZE+150+100;

// colors
const SDL_Color RED = {240, 101, 67, 255};
const SDL_Color YELLOW = {255, 200, 87, 255};
const SDL_Color DARK_BLACK = {10, 9, 8, 255};
const SDL_Color DARK_BLUE = {4, 2, 68, 255};
const SDL_Color WHITE = {253, 244, 220, 255};
const SDL_Color GREEN = {152,251,152, 255};
const SDL_Color DARK_GREEN = {15,90,50, 255};

void set_col(SDL_Renderer* renderer, SDL_Color color);
int update_elo(int elo_1, int elo_2, int winner);

typedef struct SDL_Circle {
    int x, y, r;
} SDL_Circle;

struct connect_four_board {
    int turn;
    int turns;
    int selected_col;
    int selected_row;
    int board[6][7];
    int128 game_state;
    SDL_Circle circles[6][7];
    SDL_Rect rect;

    bool win();
    int get_row();
};

struct Player {
    int elo = 1000;
    virtual int get_col(bool train) = 0;
};

struct MCTS : public Player {
    std::map<int128, float> wins;
    std::map<int128, int> sims;

    bool random_roll_out = false;
    int num_roll_outs = 100;
    int iterations = 10000;

    float c = sqrt(2.0f);
    float discount_factor = 1; // TODO: this is not functional yet

    int get_col(bool train);

    float UCT(int128 v, int128 p);
    int128 get_parent(int128 v);

    void run(connect_four_board board);
    void select(connect_four_board &board);
    void expand(connect_four_board &board);
    int roll_out(connect_four_board board);
    int roll_out_rand(connect_four_board board);
    void backup(int128 game_state, float result);

    int get_best_move(connect_four_board board);

    void play(connect_four_board &board);

    void save(std::string filename = "mcts.txt");
    void load(std::string filename = "mcts.txt");
    void train(int num_games);
    std::vector<int> can_win(int player, connect_four_board board);
};

enum {
    SCREEN_CONNECT_FOUR,
    SCREEN_END,
    SCREEN_MENU
};

enum {
    MAN_N = 0,
    DQN_N = 1,
    MCTS_N = 2
};

void SDL_RenderFillCircle(SDL_Renderer* renderer, SDL_Circle* circle);

struct Screen {
    SDL_Renderer* renderer;

    TTF_Font* font;
    SDL_Surface* surface;
    SDL_Texture* texture;

    connect_four_board board;

    std::string player_1 = "MAN";
    std::string player_2 = "MAN";

    enum {
        CONTINUE = -1,
        EXIT = -2
    };

    virtual bool init() = 0;
    virtual int loop() = 0;
    virtual void close() = 0;

    bool init_all();
    void close_all();
    void display_text(const char* text, int x, int y, int size, bool show_text_field = false, int start_x = 0, int width = 0, SDL_Color col = GREEN);
};

struct Connect_four_screen : public Screen {
    int game_type;
    int cur;
    MCTS mcts_1;
    MCTS mcts_2;

    Connect_four_screen(int status) : game_type(status) {};

    bool init();
    int loop();
    void close();

    void render_board();
    void falling();
    void pick_col(int col);
    int play();
    int DQN();
    int MCTS_func();
};

struct End_screen : public Screen {
    int winner;
    int x, y;
    // winner gets set to status
    End_screen(int status, int x, int y) : winner(status), x(x), y(y) {};

    bool init();
    int loop();
    void close();
};

struct Menu_screen : public Screen {
    std::vector<int> selected;
    bool cur_col = false;

    std::vector<std::string> text = {"MAN", "DQN", "MCTS"};

    Menu_screen(std::vector<int> selected) : selected(selected) {};

    bool init();
    int loop();
    void close();

    void render_screen();
    std::string get_text();
    int mode();
};

Screen* switch_screen(Screen* screen, int new_screen, int status = 0);

#endif //CONNECT_FOUR_GAME_H
