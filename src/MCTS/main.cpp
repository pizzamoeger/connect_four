#include "../game.h"

int main(int argc, char* argv[]) {
    srand(time(NULL));

    if (argc > 1) {
        // check if there are 4 arguments
        if (argc != 5) {
            std::cerr << "Invalid commandline arguments. Please check README.md\n";
            return 0;
        }

        // check if first argument is 0 or 1
        if (argv[1][0] != '0' && argv[1][0] != '1') {
            std::cerr << "Invalid commandline arguments. Please check README.md\n";
            return 0;
        }

        // check if all the other arguments are ints
        for (int i = 2; i < 5; i++) {
            for (char c : std::string(argv[i])) {
                if (c < '0' || c > '9') {
                    std::cerr << "Invalid commandline arguments. Please check README.md\n";
                    return 0;
                }
            }
        }

        std::cerr << "training MCTS...\n";
        MCTS mcts;

        mcts.random_simulation = atoi(argv[1]);
        mcts.simulations = atoi(argv[2]);
        mcts.iterations = atoi(argv[3]);
        int num_games = atoi(argv[4]);

        mcts.training = true;
        mcts.train(num_games);

        std::string filename = "data/MCTS/";
        if (mcts.random_simulation) filename += "r_";

        filename += std::to_string(mcts.simulations);
        filename += "_"+std::to_string(mcts.iterations);
        filename += "_"+std::to_string(num_games);
        filename += ".txt";

        mcts.save(filename);
        std::cerr << "saved MCTS at " << filename << "\n";
        return 0;
    }

    std::cerr << "Invalid commandline arguments. Please check README.md\n";
    return 0;
}
