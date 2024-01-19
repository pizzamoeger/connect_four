#include "../game.h"
#include "SDL.h"

// memleaks: valgrind --leak-check=full --show-leak-kinds=all cmake-build-release/connect_four
std::shared_ptr<Screen> switch_screen(std::shared_ptr<Screen> screen, int new_screen, int status) {
    screen->close();

    int x = 0;
    int y = 0;

    //SDL_Window* tmp_window = screen->window;
    SDL_Renderer* tmp_renderer = screen->renderer;
    SDL_connect_four_board tmp_board = screen->board;
    std::string tmp_player_1 = screen->playerfile_1;
    std::string tmp_player_2 = screen->playerfile_2;

    switch (new_screen) {
        case SCREEN_CONNECT_FOUR:
            screen = std::make_shared<Connect_four_screen>(status);
            break;

        case SCREEN_END:
            x = (SCREEN_WIDTH - 800) / 2 + 800 / 2;
            y = (SCREEN_HEIGHT - (700-150+TEXT_SIZE+TEXT_DIST)) / 2 + 700 + 70;

            if (status == 2) status = -1;
            screen = std::make_shared<End_screen>(status, x, y);
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

SDL_connect_four_board::SDL_connect_four_board() {
    game_state = 0;
    turn = 1;
    turns = 0;
    selected_row = 5;
    selected_col = 0;

    int x = (SCREEN_WIDTH - 800) / 2;
    int y = (SCREEN_HEIGHT - (700-150+TEXT_SIZE+TEXT_DIST)) / 2; // 150 is the offset from the top

    rect = { x, y, 800, 700 };
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 7; j++){
            board[i][j] = 0;
            circles[i][j] = { x+ 100 + 100 * j, y + 100 + 100 * i, 40 };
        }
    }
}

int main() {
    // init random seed
    srand(time(NULL));

    std::vector<int> sel = {0, 0};
    std::shared_ptr<Screen> screen = std::make_shared<Menu_screen>(sel);

    if (!screen->init_all()) return 0;

    // wait for user to select what mode
    int status = screen->loop();
    while(status == screen->CONTINUE) {
        status = screen->loop();
    }

    if (status == screen->EXIT) {
        screen->close_all();
        return 0;
    }

    int game_state = status; // so that game can be restarted

    while (true) {
        // clear screen
        SDL_RenderClear(screen->renderer);
        set_col(screen->renderer, WHITE);
        SDL_RenderPresent(screen->renderer);

        // switch to connect four screen
        screen = switch_screen(screen, SCREEN_CONNECT_FOUR, game_state);

        // game loop
        status = screen->CONTINUE;
        while (status == screen->CONTINUE) status = screen->loop();

        if (status == screen->EXIT) {
            screen->close_all();
            return 0;
        }

        screen = switch_screen(screen, SCREEN_END, status);

        status = screen->CONTINUE;
        while (status == screen->CONTINUE) status = screen->loop();
        if (status == screen->EXIT) break;

        // start alternating between players
        swap(screen->playerfile_1, screen->playerfile_2);
        game_state = (game_state%SELECTIONS)*SELECTIONS + game_state/SELECTIONS;
    }
    screen->close_all();
    return 0;
}
