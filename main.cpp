#include "game.h"

// run this:  valgrind --leak-check=full cmake-build-debug/connect_four  to die faster :D

int main(int argc, char* argv[]) {
    // init random seed
    srand(time(NULL));

    if (argc > 1) { // train mcts
        // check if there are 4 arguments
        if (argc != 5) {
            std::cerr << "usage: ./connect_four <random_roll_out (1/0)> <num_roll_outs> <iterations> <num_games>\n";
            return 0;
        }

        // check if first argument is 0 or 1
        if (argv[1][0] != '0' && argv[1][0] != '1') {
            std::cerr << "usage: ./connect_four <random_roll_out (1/0)> <num_roll_outs> <iterations> <num_games>\n";
            return 0;
        }

        // check if all the other arguments are ints
        for (int i = 2; i < 5; i++) {
            for (char c : std::string(argv[i])) {
                if (c < '0' || c > '9') {
                    std::cerr << "usage: ./connect_four <random_roll_out (1/0)> <num_roll_outs> <iterations> <num_games>\n";
                    return 0;
                }
            }
        }

        std::cerr << "training mcts\n";
        MCTS mcts;

        mcts.random_roll_out = atoi(argv[1]);
        mcts.num_roll_outs = atoi(argv[2]);
        mcts.iterations = atoi(argv[3]);
        int num_games = atoi(argv[4]);

        mcts.training = true;
        mcts.train(num_games);

        std::string filename = "MCTS/";
        if (mcts.random_roll_out) filename += "r_";

        filename += std::to_string(mcts.num_roll_outs);
        filename += "_"+std::to_string(mcts.iterations);
        filename += "_"+std::to_string(num_games);
        filename += ".txt";

        mcts.save(filename);
        return 0;
    }

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
