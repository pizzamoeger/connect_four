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
};

struct MCTS {
    std::map<int128, int> wins;
    std::map<int128, int> sims;
    float c;
    float gamma;

    float UCT(int128 v, int128 p);
    int128 get_parent(int128 v);

    void run(int num_roll_outs, connect_four_board board);
    void select(connect_four_board &board);
    void expand(connect_four_board &board);
    int roll_out(connect_four_board board);
    void backup(int128 game_state, float result); // result either 1 (w), 0 (t), or -1 (l) ?

    int get_best_move(connect_four_board board);

    void play(connect_four_board &board);

    void save(std::string filename = "mcts.txt");
    void load();
    void train(int num_roll_outs, int num_games);
};

enum {
    SCREEN_CONNECT_FOUR,
    SCREEN_END,
    SCREEN_MENU
};

void SDL_RenderFillCircle(SDL_Renderer* renderer, SDL_Circle* circle);

struct Screen {
    SDL_Renderer* renderer;

    TTF_Font* font;
    SDL_Surface* surface;
    SDL_Texture* texture;

    connect_four_board board;

    virtual bool init() = 0;
    virtual int loop() = 0;
    virtual void close() = 0;

    bool init_all();
    void close_all();
    void display_text(const char* text, int x, int y, int size, bool show_text_field = false, int start_x = 0, int width = 0, SDL_Color col = GREEN);
};

struct Connect_four_screen : public Screen {
    int AI_player;
    MCTS mcts;
    Connect_four_screen(int AI_player) : AI_player(AI_player) {
        mcts.c = 1.0f / sqrt(2.0f); // TODO: this is just something github copilot suggested
    };

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
    int selected;
    Menu_screen(int selected) : selected(selected) {};

    bool init();
    int loop();
    void close();

    void render_screen();
};

#endif //CONNECT_FOUR_GAME_H
