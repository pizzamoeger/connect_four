#include "game.h"

Screen* switch_screen(Screen* screen, int new_screen, int status) {
    screen->close();

    int x = 0;
    int y = 0;

    //SDL_Window* tmp_window = screen->window;
    SDL_Renderer* tmp_renderer = screen->renderer;
    connect_four_board tmp_board = screen->board;
    std::string tmp_player_1 = screen->playerfile_1;
    std::string tmp_player_2 = screen->playerfile_2;

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
    std::swap(screen->playerfile_1, tmp_player_1);
    std::swap(screen->playerfile_2, tmp_player_2);

    screen->init();
    return screen;
}

std::pair<float,float> update_elo(float elo_1, float elo_2, int result) {
    const int K = 32;
    double expected_1 = 1.0 / (1.0 + pow(10.0, (elo_2 - elo_1) / 400.0));
    double expected_2 = 1.0 / (1.0 + pow(10.0, (elo_1 - elo_2) / 400.0));

    // TODO: could be written nicer
    float score_1;
    float score_2;
    if (result == 1) {
        score_1 = 0;
        score_2 = 1;
    } else if (result == -1) {
        score_1 = 1;
        score_2 = 0;
    } else {
        score_2 = 0.5;
        score_1 = 0.5;
    }

    int new_elo_1 = elo_1 + K * (score_1 - expected_1);
    int new_elo_2 = elo_2 + K * (score_2 - expected_2);

    return {new_elo_1, new_elo_2};
}