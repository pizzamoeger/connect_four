#include "game.h"

// TODO : i want to die. i9dk wo zum fick memleak bruh
// run this:  valgrind --leak-check=full cmake-build-debug/connect_four  to die faster :D

Screen* switch_screen(Screen* screen, int new_screen, int status = 0) {
    screen->close();

    int x = 0;
    int y = 0;

    //SDL_Window* tmp_window = screen->window;
    SDL_Renderer* tmp_renderer = screen->renderer;
    connect_four_board tmp_board = screen->board;

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

    screen->init();
    return screen;
}

int main(int argc, char* argv[]) {
    // init random seed
    srand(time(NULL));

    if (argc > 1) { // train mcts
        std::cerr << "training mcts\n";
        int num_games = atoi(argv[1]);
        MCTS mcts;
        mcts.train(num_games);
        if (argc > 2) mcts.save(argv[2]);
        else mcts.save();
        return 0;
    }

    Screen* screen = new Menu_screen({0, 0});
    // init random
    srand(time(NULL));

    if (!screen->init_all()) {
        return 0;
    }

    // wait for user to select what mode
    int status = screen->CONTINUE;
    while(status == screen->CONTINUE) {
        status = screen->loop();
    }

    if (status == screen->EXIT) {
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
    status = screen->CONTINUE;
    while(status == screen->CONTINUE) status = screen->loop();

    if (status == screen->EXIT) {
        screen->close_all();
        return 0;
    }

    screen = switch_screen(screen, SCREEN_END, status);

    while (screen->loop() == screen->CONTINUE);

    screen->close_all();
    delete screen;

    return 0;
}
