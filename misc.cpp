#include "game.h"

Screen* switch_screen(Screen* screen, int new_screen, int status) {
    screen->close();

    int x = 0;
    int y = 0;

    //SDL_Window* tmp_window = screen->window;
    SDL_Renderer* tmp_renderer = screen->renderer;
    connect_four_board tmp_board = screen->board;
    std::string tmp_player_1 = screen->player_1;
    std::string tmp_player_2 = screen->player_2;

    switch (new_screen) {
        case SCREEN_CONNECT_FOUR:
            delete screen;
            screen = new Connect_four_screen(status);
            break;

        case SCREEN_END:
            x = (SCREEN_WIDTH - 800) / 2 + 800 / 2;
            y = (SCREEN_HEIGHT - (700-150+TEXT_SIZE+TEXT_DIST)) / 2 + 700 + 70;

            delete screen;
            if (status == 2) status = -1;
            screen = new End_screen(status, x, y);
            break;

        default:
            break;
    }

    //std::swap(screen->window, tmp_window);
    std::swap(screen->renderer, tmp_renderer);
    std::swap(screen->board, tmp_board);
    std::swap(screen->player_1, tmp_player_1);
    std::swap(screen->player_2, tmp_player_2);

    screen->init();
    return screen;
}