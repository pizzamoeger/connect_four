#ifndef CONNECT_FOUR_SDL_H
#define CONNECT_FOUR_SDL_H

#include "../game.h"

#define DELAY 1000/60

//Screen dimension constants
const int TEXT_SIZE = 60;
const int TEXT_DIST = 100;

// FIND-TAG-DIMENSIONS
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
std::pair<float,float> update_elo(float elo_1, float elo_2, int result);

typedef struct SDL_Circle {
    int x, y, r;
} SDL_Circle;

struct SDL_connect_four_board : public connect_four_board {
    SDL_Circle circles[6][7];
    SDL_Rect rect;
};


enum {
    SCREEN_CONNECT_FOUR,
    SCREEN_END,
    SCREEN_MENU
};

enum {
    MAN = 0,
    AI = 1
};

void SDL_RenderFillCircle(SDL_Renderer* renderer, SDL_Circle* circle);

struct Screen {
    SDL_Renderer* renderer;

    TTF_Font* font;
    SDL_Surface* surface;
    SDL_Texture* texture;

    SDL_connect_four_board board;

    std::string playerfile_1 = "HUMAN/test.txt";
    std::string playerfile_2 = "HUMAN/test.txt";

    enum {
        CONTINUE = -1,
        EXIT = -2
    };

    virtual ~Screen() = default;
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
    std::unique_ptr<Player> player_1;
    std::unique_ptr<Player> player_2;

    Connect_four_screen(int status) : game_type(status) {};

    bool init();
    int loop();
    void close();

    void render_board();
    void falling();
    void pick_col(int col);
    int play();
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

const int SELECTIONS = 2;

struct Menu_screen : public Screen {
    std::vector<int> selected;
    bool cur_col = false;

    std::vector<std::string> text = {"MAN", "AI"};

    Menu_screen(std::vector<int> selected) : selected(selected) {};

    bool init();
    int loop();
    void close();

    void render_screen();
    std::string get_text(std::string what = "ENTER FILENAME");
    int mode();
};

std::shared_ptr<Screen> switch_screen(std::shared_ptr<Screen> screen, int new_screen, int status = 0);

#endif //CONNECT_FOUR_SDL_H
