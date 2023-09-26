#ifndef CONNECT_FOUR_GAME_H
#define CONNECT_FOUR_GAME_H

#include <SDL2/SDL.h>
#include <iostream>
#include <cmath>
#include <SDL2/SDL_ttf.h>
#include <vector>

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
    SDL_Circle circles[6][7];
    SDL_Rect rect;
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
    Connect_four_screen(int AI_player) : AI_player(AI_player) {};

    bool init();
    int loop();
    void close();

    bool win();
    void render_board();
    void falling();
    int play();
    int DQN();
    int MCTS();
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

struct MCTS {
    std::vector<int> values;
    std::vector<int> visits;
    std::vector<std::vector<int>> graph;
    int c;

    float UCB(int v, int p);
};

#endif //CONNECT_FOUR_GAME_H
