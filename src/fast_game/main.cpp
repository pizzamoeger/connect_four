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
        else if (playerfile[0] == 'D') player = std::make_unique<DQN>();
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

    while (true) {
        int action;
        if (board.turn%2 == 1) action = player_1->get_col(board, true);
        else action = player_2->get_col(board, true);

        // execute action
        board.selected_col = action;
        board.play();

	    std::cout << board << "\n";

        if (board.win() || board.turns == 42) break;
    }

    std::pair<float, float> new_elos = update_elo(player_1->elo, player_2->elo, board.turn);
    if (file1 != file2) {
        player_1->elo = new_elos.first;
        player_2->elo = new_elos.second;
    }

    player_1->save("data/"+file1);
    player_2->save("data/"+file2);
}
