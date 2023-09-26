#include "game.h"

// TODO : i want to die. i9dk wo zum fick memleak bruh
// run this:  valgrind --leak-check=full cmake-build-debug/connect_four  to die faster :D

Screen* switch_screen(Screen* screen, int new_screen, int status = 0) {
    screen->close();

    int x = 0;
    int y = 0;
    int AI_player = status;

    //SDL_Window* tmp_window = screen->window;
    SDL_Renderer* tmp_renderer = screen->renderer;
    connect_four_board tmp_board = screen->board;

    switch (new_screen) {
        case SCREEN_CONNECT_FOUR:
            delete screen;
            screen = new Connect_four_screen(AI_player);
            break;

        case SCREEN_END:
            x = (SCREEN_WIDTH - 800) / 2 + 800 / 2;
            y = (SCREEN_HEIGHT - (700-150+TEXT_SIZE+TEXT_DIST)) / 2 + 700 + 70;

            delete screen;
            screen = new End_screen(status, x, y);
            break;

        default:
            break;
    }

    //std::swap(screen->window, tmp_window);
    std::swap(screen->renderer, tmp_renderer);
    std::swap(screen->board, tmp_board);

    screen->init();
    return screen;
}

int main() {

    Screen* screen = new Menu_screen(0);

    if (!screen->init_all()) {
        return 0;
    }
    std::cerr << "screen created\n";

    // wait for user to select what mode
    int status = 2;
    while(status == 2) {
        status = screen->loop();
    }

    if (status == -2) {
        screen->close_all();
        return 0;
    }

    // clear screen
    SDL_RenderClear(screen->renderer);
    set_col(screen->renderer, WHITE);
    SDL_RenderPresent(screen->renderer);

    // switch to connect four screen
    screen = switch_screen(screen, SCREEN_CONNECT_FOUR, status);

    // game loop
    status = 2;
    while(status == 2) status = screen->loop();

    if (status == -2) {
        screen->close_all();
        return 0;
    }

    screen = switch_screen(screen, SCREEN_END, status);

    while (screen->loop());

    screen->close_all();
    delete screen;

    return 0;
}
