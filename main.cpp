#include "game.h"

// run this:  valgrind --leak-check=full cmake-build-debug/connect_four  to die faster :D
/*
 * SOME DOCUMENTATION:
 * you can select what plays against what on the menu screen.
 * man is manual, meaning user input
 * dqn is a neural network trained with deep q learning
 * mcts is a monte carlo tree search
 *
 * first you select how the first player will play, then you press enter.
 * now you need to enter from which file you want to load the model.
 * the only characters allowed for the filename are letters and numbers . and _
 *
 * FOR MCTS: THE FILE NEEDS TO BE OF FORMAT:
 * sims (EMPTY IF YOU WANT TO START WITH A NEW MODEL)
 * wins (EMPTY IF YOU WANT TO START WITH A NEW MODEL)
 * random_roll_out num_roll_outs iterations c discount_factor
 *
 * DQN IS NOT FUNCTIONAL YET
 * */

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
        if (status == screen->EXIT) {
            screen->close_all();
            delete screen;
            return 0;
        }

        // start alternating between players
        swap(screen->playerfile_1, screen->playerfile_2);
        game_state = (game_state%SELECTIONS)*SELECTIONS + game_state/SELECTIONS;
    }
}
