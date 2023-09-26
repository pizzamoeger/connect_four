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
            bool in_circle = 0;
            for (int row_circle = 0; row_circle < 6; row_circle++) {
                if ((x_pixel - board.circles[row_circle][board.selected_col].x) * (x_pixel - board.circles[row_circle][board.selected_col].x) +
                    (y_pixel - board.circles[row_circle][board.selected_col].y) * (y_pixel - board.circles[row_circle][board.selected_col].y) <=
                    board.circles[row_circle][board.selected_col].r * board.circles[row_circle][board.selected_col].r) {
                    in_circle = 1;
                    break;
                }
            }
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

bool Connect_four_screen::win() {
    // the last tile has beeen placed at board.selected_row, board.selected_col
    int row = board.selected_row;
    int col = board.selected_col;
    int turn = board.turn;

    // win horizontally
    for (int i = std::max(0, row-4); i < std::min(6-4+1, row+4); i++) {
        bool win = true;
        for (int j = 0; j < 4; j++) {
            if (board.board[i+j][col] != turn) {
                win = false;
                break;
            }
        }
        if (win) return true;
    }

    // win vertically
    for (int i = std::max(0, col-4); i < std::min(7-4+1, col+4); i++) {
        bool win = true;
        for (int j = 0; j < 4; j++) {
            if (board.board[row][i+j] != turn) {
                win = false;
                break;
            }
        }
        if (win) return true;
    }

    // win diagonally up left to down right
    for (int it = -3; it < 4; it++) {
        bool win = true;
        if (row + it < 0 || row + it + 3 >= 6 || col + it < 0 || col + it + 3 >= 7) continue;
        int i = row+it;
        int j = col+it;
        for (int k = 0; k < 4; k++) {
            if (board.board[i+k][j+k] != turn) {
                win = false;
                break;
            }
        }
        if (win) return true;
    }

    // win diagonally down left to up right
    for (int it = -3; it < 4; it++) {
        bool win = true;
        if (row + it < 0 || row + it + 3 >= 6 || col - it < 0 || col - it - 3 >= 7) continue;
        int i = row+it;
        int j = col-it;
        for (int k = 0; k < 4; k++) {
            if (board.board[i+k][j-k] != turn) {
                win = false;
                break;
            }
        }
        if (win) return true;
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

    return 1;
}

int Connect_four_screen::loop() { // 0: TIE, 1: player1, -1: player2, 2: continue, -2: quit
    SDL_Event event;
    int ret;

    // Events management
    while (SDL_PollEvent(&event)) {
        switch (event.type) {

            case SDL_QUIT:
                // handling of close button
                return -2;

            case SDL_KEYDOWN: {
                if (AI_player > 1) break;
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
                        if (ret != 2) {
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

    // clears the screen
    SDL_RenderClear(renderer);

    // renders board
    render_board();
    set_col(renderer, WHITE);

    // display on screen
    SDL_RenderPresent(renderer);

    // calculates to 60 fps
    SDL_Delay(1000 / 60);

    if (board.turns == 42) return 0;

    ret = 2;
    if (AI_player == board.turn) ret = DQN();
    if (AI_player == 3) {
        if (board.turn == 1) ret = DQN();
        else ret = MCTS();
    }
    if (AI_player == 4) {
        if (board.turn == 1) ret = MCTS();
        else ret = DQN();
    }

    return ret;
}

int Connect_four_screen::play() {
    while (board.board[board.selected_row][board.selected_col] != 0) {
        board.selected_row--;
        if (board.selected_row < 0) break;
    }
    if (board.selected_row >= 0) {

        // cool animation
        falling();

        // checks if the Connect_four_screen is over
        if (win()) {
            board.turn = -board.turn;
            SDL_RenderClear(renderer);
            render_board();
            SDL_RenderPresent(renderer);
            return -board.turn;
        }

        board.turns++;
        board.turn = -board.turn;
        board.selected_row = 5;
    }

    return 2;
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
}

int Connect_four_screen::DQN() {
    // THIS IS CURRENTLY STUPID AI
    int col = rand() % 7;
    while (board.board[0][col] != 0) col = rand() % 7;

    // a cool animation for selecting the column
    while (board.selected_col != col) {
        int add = 1;
        if (board.selected_col > col) add = -1;

        // it is thinking really hard
        if (rand()%5 == 0) add = -add;

        board.selected_col += add;

        SDL_RenderClear(renderer);
        render_board();

        set_col(renderer, WHITE);
        SDL_RenderPresent(renderer);

        // also thinking a lot
        SDL_Delay(rand()%900);
    }

    return play();
}

int Connect_four_screen::MCTS() {
    return DQN();
}

void Connect_four_screen::close() {
    // free memory
}

bool Screen::init_all() {
    if (!init()) {
        return 0;
    }

    // returns zero on success else non-zero
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cout << "error initializing SDL: " << SDL_GetError() << "\n";
        return 0;
    }


    if (TTF_Init() < 0) {
        std::cout << "Error initializing SDL_ttf: " << TTF_GetError() << "\n";
    }

    // renderer to render images
    renderer = SDL_CreateRenderer(SDL_CreateWindow("CONNECT FOUR",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,SCREEN_WIDTH, SCREEN_HEIGHT, 0), -1, 0);
    //exit(0);
    if (renderer == NULL) {
        std::cout << "Error initializing SDL renderer: " << SDL_GetError() << "\n";
        return 0;
    }

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
    font = TTF_OpenFont("01211_AHDSANSB.ttf", size);
    if (font == NULL) std::cout << "error loading font: " << TTF_GetError() << "\n";

    surface = TTF_RenderText_Solid(font, text, DARK_BLACK);
    if (surface == NULL) std::cout << "error creating surface: " << TTF_GetError() << "\n";

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

    if (show_text_field) {
        set_col(renderer, col);
        SDL_Rect larger_text_field = {start_x, text_field.y - 30, width, text_field.h + 2 * 30};
        SDL_RenderFillRect(renderer, &larger_text_field);
    }

    // TODO: somehow this line is causing a bugger BUT MAYBE I FIXED IT
    // and only if all the code regarding the rendering text on menu screen stays EXACTLY
    // &text_field
    SDL_RenderCopy(renderer, texture, NULL, &text_field);

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
    // get event
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
        switch (event.type) {

            case SDL_QUIT:
                // handling of close button
                return 0;

            default:
                break;
        }
    }

    return 1;
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
    // get event
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
        switch (event.type) {

            case SDL_QUIT:
                // handling of close button
                return -2;

            case SDL_KEYDOWN: {
                // keyboard API for key pressed
                switch (event.key.keysym.scancode) {

                    case SDL_SCANCODE_UP:
                        selected = std::max(0, selected - 1);
                        break;

                    case SDL_SCANCODE_DOWN:
                        selected = std::min(4, selected + 1);
                        break;

                    case SDL_SCANCODE_RETURN:
                        if (selected > 2) return selected;
                        return selected - 1;

                    default:
                        break;
                }
            }
            default:
                break;
        }
    }

    SDL_RenderClear(renderer);
    render_screen();

    set_col(renderer, WHITE);
    SDL_RenderPresent(renderer);

    return 2;
}

void Menu_screen::render_screen() {
    int x = 275;
    int w = 450;

    std::vector<std::string> text = {"YOU START", "2 PLAYER", "AI STARTS", "DQN VS MCTS", "MCTS VS DQN"};
    std::vector<int> y_pos = {SCREEN_HEIGHT/6, 2*SCREEN_HEIGHT/6, 3*SCREEN_HEIGHT/6, 4*SCREEN_HEIGHT/6, 5*SCREEN_HEIGHT/6};

    for (int i = 0; i < 5; i++) {
        if (i == selected) {
            display_text(text[i].c_str(), -1, y_pos[i], TEXT_SIZE, 1, x, w, DARK_GREEN);
            continue;
        }
        else display_text(text[i].c_str(), -1, y_pos[i], TEXT_SIZE, 1, x, w, GREEN);
    }
}