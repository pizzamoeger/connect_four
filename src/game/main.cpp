#include "game.h"

// memleaks: valgrind --leak-check=full --show-leak-kinds=all cmake-build-release/connect_four

int main() {
    // init random seed
    srand(time(NULL));

    std::vector<int> sel = {0, 0};
    std::shared_ptr<Screen> screen = std::make_shared<Menu_screen>(sel);

    if (!screen->init_all()) { // TODO ???
        return 0;
    }

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

    int counter = 0; // COUNTER IS FOR GETTING INITIAL ELOS
    // FIND-TAG-COUNTER
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
        if (status == screen->EXIT) {
            screen->close_all();
            return 0;
        }

        // start alternating between players
        swap(screen->playerfile_1, screen->playerfile_2);
        game_state = (game_state%SELECTIONS)*SELECTIONS + game_state/SELECTIONS;

        counter++;
        // std::cerr << counter << "\n";
    }
    screen->close_all();
    return 0;
}
