#include "../game.h"

int main() {
    // init random seed
    srand(time(NULL));

    connect_four_board board;

    auto init_player = [&] (std::string playerfile) {
        std::unique_ptr<Player> player = nullptr;

        if (playerfile.size() == 0) playerfile = 'i'; // invalid and error will be printed

        if (playerfile[0] == 'R') player = std::make_unique<Random>();
        else if (playerfile[0] == 'M') player = std::make_unique<MCTS>();
        else if (playerfile[0] == 'D') player = std::make_unique<DQL>();
        else if (playerfile[0] == 'A') player = std::make_unique<Almost_random>();
        else if (playerfile[0] == 'H') player = std::make_unique<Human>();
        else {
            std::cerr << "Error: invalid playerfile. ";
            playerfile = "Random Bot";
            player = std::make_unique<Random>();
        }
        //std::cerr << "loaded " << playerfile << "\n";
        player->load("data/"+playerfile);
        return player;
    };

    std::string file1; std::cin >> file1;
    std::unique_ptr<Player> player_1;
    player_1 = std::move(init_player(file1));
    std::string file2; std::cin >> file2;
    std::unique_ptr<Player> player_2;
    player_2 = std::move(init_player(file2));

    int DQL_PLAYER; std::cin >> DQL_PLAYER;

    while (true) {
        int action;
        if (board.turn%2 == 1) action = player_1->get_col(board);
        else action = player_2->get_col(board);

        // execute action
        board.selected_col = action;
        board.play();
        //if (action == 2) break;

        /*for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 7; j++) {
                if (board.board[i][j] == -1) std::cerr << "x";
                else if (board.board[i][j] == 1) std::cerr << "o";
                else std::cerr << "_";
                std::cerr << " ";
            }
            std::cerr << "\n";
        }
        std::cerr << "\n\n\n";*/

        if (board.win() || board.turns == 42) break;
    }

    std::cout << (board.turn==DQL_PLAYER?-1:1) << "\n";
}