// this code is built onto the tutorial https://www.geeksforgeeks.org/sdl-library-in-c-c-with-examples/

#include "game.h"

void set_col(SDL_Renderer* renderer, SDL_Color color) {
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
}

void SDL_RenderFillCircle(SDL_Renderer* renderer, SDL_Circle* circle) {
    // this code was kindly provided by Github copilot
    int offsetx, offsety, d;

    offsetx = 0;
    offsety = circle->r;
    d = circle->r - 1;
    while (offsety >= offsetx) {
        SDL_RenderDrawLine(renderer, circle->x - offsety, circle->y + offsetx, circle->x + offsety, circle->y + offsetx);
        SDL_RenderDrawLine(renderer, circle->x - offsetx, circle->y + offsety, circle->x + offsetx, circle->y + offsety);
        SDL_RenderDrawLine(renderer, circle->x - offsetx, circle->y - offsety, circle->x + offsetx, circle->y - offsety);
        SDL_RenderDrawLine(renderer, circle->x - offsety, circle->y - offsetx, circle->x + offsety, circle->y - offsetx);

        if (d >= 2 * offsetx) {
            d -= 2 * offsetx + 1;
            offsetx += 1;
        } else if (d < 2 * (circle->r - offsety)) {
            d += 2 * offsety - 1;
            offsety -= 1;
        } else {
            d += 2 * (offsety - offsetx - 1);
            offsety -= 1;
            offsetx += 1;
        }
    }
}

void Connect_four_screen::render_board() {
    // renders the board
    for (int x_pixel = board.rect.x; x_pixel < board.rect.x + board.rect.w; x_pixel++) {
        for (int y_pixel = board.rect.y; y_pixel < board.rect.y + board.rect.h; y_pixel++) {
            // iterates over the circles vector and checks if the pixel is in the circle
            bool in_circle = false;
            for (int row_circle = 0; row_circle < 6; row_circle++) {
                if ((x_pixel - board.circles[row_circle][board.selected_col].x) * (x_pixel - board.circles[row_circle][board.selected_col].x) +
                    (y_pixel - board.circles[row_circle][board.selected_col].y) * (y_pixel - board.circles[row_circle][board.selected_col].y) <=
                    board.circles[row_circle][board.selected_col].r * board.circles[row_circle][board.selected_col].r) {
                    in_circle = true;
                    break;
                }
            }
            // color the pixel if it is not in the circle
            if (!in_circle) {
                set_col(renderer, DARK_BLUE);
                SDL_RenderDrawPoint(renderer, x_pixel, y_pixel);
            }
        }
    }

    // renders the piece at the top
    if (board.turn == 1) set_col(renderer, RED);
    else set_col(renderer, YELLOW);
    SDL_Circle circle = {board.circles[0][board.selected_col].x, board.rect.y - 70, 40 };
    SDL_RenderFillCircle(renderer, &circle);

    // renders all the circles
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 7; j++) {
            if (j == board.selected_col && board.board[i][j] == 0) continue;
            if (board.board[i][j] == 1) set_col(renderer, RED);
            else if (board.board[i][j] == -1) set_col(renderer, YELLOW);
            else set_col(renderer, WHITE);
            SDL_RenderFillCircle(renderer, &board.circles[i][j]);
        }
    }
}

bool connect_four_board::win() {
    // place of last tile
    int placed_row = get_row()+1;
    int placed_col = selected_col;
    int turn = board[placed_row][placed_col];

    // length of continuous pattern of correct tile in dis xstp ystp
    auto check = [&](int xstp, int ystp) {
        int col = placed_col, row = placed_row;
        while (0 <= col && col < 7 && 0 <= row && row < 6 && board[row][col] == turn) {
            col += xstp;
            row += ystp;
        }
        return std::max(abs(col - placed_col), abs(row - placed_row)); // length
    };

    // look for all possible patterns
    for (int xstp: {0, 1}) {
        for (int ystp: {-1, 0, 1}) {
            if (xstp == 0 && ystp == 0) continue;
            if (check(xstp, ystp) + check(-xstp, -ystp) >= 5) return true;
        }
    }
    return false;
}

bool Connect_four_screen::init() {

    // init board
    int x = (SCREEN_WIDTH - 800) / 2;
    int y = (SCREEN_HEIGHT - (700-150+TEXT_SIZE+TEXT_DIST)) / 2; // 150 is the offset from the top

    board.turn = 1;
    board.turns = 0;

    board.selected_col = 0;
    board.selected_row = 5;

    board.rect = { x, y, 800, 700 };
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 7; j++){
            board.board[i][j] = 0;
            board.circles[i][j] = { x+ 100 + 100 * j, y + 100 + 100 * i, 40 };
        }
    }

    // init player
    auto init_player = [&] (std::string playerfile, Player* player) {
        std::cerr << "loading " << playerfile << "\n";
        assert(playerfile.size() > 0);

        if (playerfile[0] == 'R') {
            player = new Random();
            return player;
        }
        if (playerfile[0] == 'M') {
            player = new MCTS();
            return player;
        }
        if (playerfile[0] == 'D') {
            player = new DQN();
            return player;
        }
        if (playerfile[0] == 'A') {
            player = new Almost_random();
            return player;
        }
        if (playerfile[0] == 'H') {
            player = new Human();
            return player;
        }
        std::cerr << "error: invalid playerfile, loaded random bot\n";
        player = new Random();
        return player;
    };

    player_1 = init_player(playerfile_1, player_1);
    player_2 = init_player(playerfile_2, player_2);

    player_1->load(playerfile_1);
    player_2->load(playerfile_2);

    return 1;
}

int Connect_four_screen::loop() {
    SDL_Event event;
    int ret;
    if (board.turn == 1) cur = game_type / SELECTIONS;
    else cur = game_type%SELECTIONS;

    // Events management
    while (SDL_PollEvent(&event)) {
        switch (event.type) {

            case SDL_QUIT:
                // handling of close button
                return EXIT;

            case SDL_KEYDOWN: {
                if (cur == AI) break;
                // keyboard API for key pressed
                switch (event.key.keysym.scancode) {

                    case SDL_SCANCODE_LEFT:
                        board.selected_col = std::max(0, board.selected_col - 1);
                        break;

                    case SDL_SCANCODE_RIGHT:
                        board.selected_col = std::min(6, board.selected_col + 1);
                        break;

                    case SDL_SCANCODE_RETURN:
                        ret = play();
                        if (ret != CONTINUE) {
                            return ret;
                        }
                        break;

                    default:
                        break;
                }
            }

            default:
                break;
        }
    }

    // updates screen
    SDL_RenderClear(renderer);
    render_board();
    set_col(renderer, WHITE);
    SDL_RenderPresent(renderer);

    // calculates to 60 fps
    SDL_Delay(1000 / 60);

    if (board.turns == 42) { // tie
        // calculate new elo
        std::pair<int,int> new_elos = update_elo(player_1->elo, player_2->elo, 0);
        player_1->elo = new_elos.first;
        player_2->elo = new_elos.second;

        // save
        player_1->save(playerfile_1);
        player_2->save(playerfile_2);

        return 0;
    }

    ret = CONTINUE;

    if (cur == MAN) return ret;

    // current player is AI
    int col;
    if (board.turn == 1) col = player_1->get_col(board);
    else col = player_2->get_col(board);
    pick_col(col);
    ret = play();

    return ret;
}

int Connect_four_screen::play() {
    board.selected_row = board.get_row();
    if (board.selected_row >= 0) {

        // animation for falling of tile
        falling();

        // checks game is over
        if (board.win()) { // TODO here elo
            board.turn = -board.turn;
            SDL_RenderClear(renderer);
            render_board();
            SDL_RenderPresent(renderer);

            // calculate new elo
            std::pair<int,int> new_elos = update_elo(player_1->elo, player_2->elo, board.turn);
            player_1->elo = new_elos.first;
            player_2->elo = new_elos.second;

            // TODO: only save when prompted
            player_1->save(playerfile_1);
            player_2->save(playerfile_2);

            if (board.turn == 1) return 2; // second player has won
            return 1; // first player has won
        }

        board.turns++;
        board.turn = -board.turn;
        board.selected_row = 5;
    }

    return CONTINUE;
}

void Connect_four_screen::falling() {

    // animation of the piece falling
    SDL_Circle falling_circle = {board.circles[board.selected_row][board.selected_col].x, board.rect.y - 70, 40 };
    while (falling_circle.y < board.circles[board.selected_row][board.selected_col].y) {
        falling_circle.y += 40;

        SDL_RenderClear(renderer);

        if (board.turn == 1) set_col(renderer, RED);
        else set_col(renderer, YELLOW);
        SDL_RenderFillCircle(renderer, &falling_circle);

        render_board();

        set_col(renderer, WHITE);
        SDL_RenderPresent(renderer);
    }

    // updates the board
    board.board[board.selected_row][board.selected_col] = board.turn;
    board.game_state = 7*board.game_state+board.selected_col+1;
}

void Connect_four_screen::close() {
    // delete players
    delete player_1;
    delete player_2;
}

void Connect_four_screen::pick_col(int col) {
    // animation for selecting column
    while (board.selected_col != col) {
        int add = 1;
        if (board.selected_col > col) add = -1;

        // it is thinking really hard
        //if (rand()%5 == 0) add = -add;

        board.selected_col += add;
        if (board.selected_col < 0) board.selected_col = 0;

        SDL_RenderClear(renderer);
        render_board();

        set_col(renderer, WHITE);
        SDL_RenderPresent(renderer);

        //SDL_Delay(rand()%900);
        SDL_Delay(200);
    }
}

bool Screen::init_all() {
    if (!init()) {
        return 0;
    }

    // init SDL
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cout << "Error initializing SDL: " << SDL_GetError() << "\n";
        return 0;
    }

    // init SDL_ttf
    if (TTF_Init() < 0) {
        std::cout << "Error initializing SDL_ttf: " << TTF_GetError() << "\n";
    }

    // renderer to render images
    renderer = SDL_CreateRenderer(SDL_CreateWindow("CONNECT FOUR",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,SCREEN_WIDTH, SCREEN_HEIGHT, 0), -1, 0);
    if (renderer == NULL) {
        std::cout << "Error initializing SDL renderer: " << SDL_GetError() << "\n";
        return 0;
    }

    // set background color
    set_col(renderer, WHITE);
    SDL_RenderClear(renderer);

    return 1;
}

void Screen::close_all() {
    close();

    // destroy renderer
    SDL_DestroyRenderer(renderer);

    // close SDL_ttf
    TTF_Quit();

    // close SDL
    SDL_Quit();
}

void Screen::display_text(const char* text, int x, int y, int size, bool show_text_field, int start_x, int width, SDL_Color col) {
    // get font
    font = TTF_OpenFont("01211_AHDSANSB.ttf", size);
    if (font == NULL) std::cout << "error loading font: " << TTF_GetError() << "\n";

    // create surface
    surface = TTF_RenderText_Solid(font, text, DARK_BLACK);
    if (surface == NULL) std::cout << "error creating surface: " << TTF_GetError() << "\n";

    // create texture from surface
    texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (texture == NULL) std::cout << "error creating texture: " << TTF_GetError() << "\n";

    int w = surface->w;
    int h = surface->h;

    SDL_Rect text_field;
    if (x == -1) x = SCREEN_WIDTH / 2;
    if (y == -1) y = SCREEN_HEIGHT / 2;

    text_field.x = x - w / 2;
    text_field.y = y - h / 2;
    text_field.w = w;
    text_field.h = h;

    // display text field
    if (show_text_field) {
        set_col(renderer, col);
        SDL_Rect larger_text_field = {start_x, text_field.y - 30, width, text_field.h + 2 * 30};
        SDL_RenderFillRect(renderer, &larger_text_field);
    }

    // display text
    SDL_RenderCopy(renderer, texture, NULL, &text_field);

    // close and free
    SDL_DestroyTexture(texture);
    TTF_CloseFont(font);
    SDL_FreeSurface(surface);
}

bool End_screen::init() {
    if (winner == 0) display_text("TIE", x, y, TEXT_SIZE);
    else if (winner == 1) display_text("RED WINS", x, y, TEXT_SIZE);
    else if (winner == -1) display_text("YELLOW WINS", x, y, TEXT_SIZE);

    set_col(renderer, WHITE);
    SDL_RenderPresent(renderer);

    return 1;
}

int End_screen::loop() {
    SDL_Event event;

    // handles events
    while (SDL_PollEvent(&event)) {
        switch (event.type) {

            case SDL_QUIT:
                return EXIT;

            case SDL_KEYDOWN:
                return 0;

            default:
                break;
        }
    }

    return CONTINUE;
}

void End_screen::close() {
    return;
}

bool Menu_screen::init() {
    return 1;
}

void Menu_screen::close() {
    return;
}

int Menu_screen::loop() {
    SDL_Event event;

    // handles events
    while (SDL_PollEvent(&event)) {
        switch (event.type) {

            case SDL_QUIT:
                return EXIT;

            case SDL_KEYDOWN: {
                switch (event.key.keysym.scancode) {

                    case SDL_SCANCODE_UP:
                        selected[cur_col] = std::max(0, selected[cur_col] - 1);
                        break;

                    case SDL_SCANCODE_DOWN:
                        selected[cur_col] = std::min(SELECTIONS - 1, selected[cur_col] + 1);
                        break;

                    case SDL_SCANCODE_RETURN:
                        if (selected[cur_col] == AI) {
                            if (cur_col == 0) playerfile_1 = get_text();
                            else playerfile_2 = get_text();

                            if (playerfile_1 == EXIT_STR || playerfile_2 == EXIT_STR) return EXIT;
                        } else {
                            if (cur_col == 0) playerfile_1 = get_text("ENTER NAME");
                            else playerfile_2 = get_text("ENTER NAME");

                            if (playerfile_1 == EXIT_STR || playerfile_2 == EXIT_STR) return EXIT;
                        }
                        if (cur_col) {
                            return mode();
                        }
                        cur_col = 1;

                    default:
                        break;
                }
            }
            default:
                break;
        }
    }

    // updates screen
    SDL_RenderClear(renderer);
    render_screen();
    set_col(renderer, WHITE);
    SDL_RenderPresent(renderer);

    return CONTINUE;
}

std::string Menu_screen::get_text(std::string what) {
    SDL_Event event;
    std::string text = "";
    bool quit = false;

    // handles events
    while (!quit) {
        //std::cout << text << "\n";
        while (SDL_PollEvent(&event)) {
            switch (event.type) {

                case SDL_QUIT:
                    return EXIT_STR;

                case SDL_KEYDOWN: {
                    switch (event.key.keysym.scancode) {

                        case SDL_SCANCODE_RETURN:
                            quit = true;
                            break;

                        case SDL_SCANCODE_BACKSPACE:
                            if (text.size() > 0) text.pop_back();
                            break;

                        default:
                            break;
                    }
                }

                case SDL_TEXTINPUT:
                    // the only characters allowed are letters and numbers . / and _
                    if (event.text.text[0] >= 'a' && event.text.text[0] <= 'z') text += event.text.text;
                    if (event.text.text[0] >= 'A' && event.text.text[0] <= 'Z') text += event.text.text;
                    if (event.text.text[0] >= '0' && event.text.text[0] <= '9') text += event.text.text;
                    if (event.text.text[0] == '.') text += event.text.text;
                    if (event.text.text[0] == '_') text += event.text.text;
                    if (event.text.text[0] == '/') text += event.text.text;

                    break;

                default:
                    break;
            }
        }

        // updates screen
        SDL_RenderClear(renderer);
        display_text(what.c_str(), -1, SCREEN_HEIGHT/4, TEXT_SIZE, 0, 0, SCREEN_WIDTH, DARK_GREEN);
        if (text.size() != 0) display_text(text.c_str(), -1, -1, TEXT_SIZE, 0, 0, SCREEN_WIDTH, DARK_GREEN);
        set_col(renderer, WHITE);
        SDL_RenderPresent(renderer);

        // calculates to 60 fps
        SDL_Delay(1000 / 60);
    }

    std::cerr << text << "\n";
    if (what == "ENTER NAME") text = "HUMAN/"+text+".txt";

    return text;
}

void Menu_screen::render_screen() {
    int w = 250;

    //std::vector<int> y_pos = {SCREEN_HEIGHT/4, 2*SCREEN_HEIGHT/4, 3*SCREEN_HEIGHT/4};

    // display buttons
    for (int col = 0; col < 2; col++) {
        int x = (col+1)*2*SCREEN_WIDTH/6 - 125;
        for (int button = 0; button < text.size(); button++) {
            int y_pos = (button+1)*SCREEN_HEIGHT/(text.size()+1);
            if (button == selected[col]) {
                display_text(text[button].c_str(), (col+1)*2*SCREEN_WIDTH/6, y_pos, TEXT_SIZE, 1, x, w, DARK_GREEN);
                continue;
            }
            else display_text(text[button].c_str(), (col+1)*2*SCREEN_WIDTH/6, y_pos, TEXT_SIZE, 1, x, w, GREEN);
        }
    }
    // TODO: maybe vs in middle
}

int Menu_screen::mode() {
    return selected[0] * SELECTIONS + selected[1];
}

int connect_four_board::get_row() { // return -1 if invalid
    int row = 5;
    while (board[row][selected_col] != 0) {
        row--;
        if (row < 0) break;
    }
    return row;
}