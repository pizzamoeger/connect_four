# Connect Four

## Description
This is a C++ Program for playing Connect four. It is written in SDL2.
You can play against a friend or against the computer.

## Requirements
You have to have access SDL2, SDL_ttf and boost multiprecision.

## Compiling and running
To compile, you can simply run `cmake CmakeList.txt` and then `make`.
To run, you can run `./connect_four`.
You can also train an MCTS. To do so, run `./connect_four <num games> <filename where it will be saved>`. Hyperparameters can be adjusted directly in game.h in the MCTS struct.

## Controls
If you ran the program without any arguments, a window will open. You can select the first player (manual or computer) by using the arrow key and then pressing enter. Now you will either be prompted for your name, which is pretty straight forward, or the file in which your opponent is stored. For this you have the following options:
* Random: enter `RANDOM/bot.txt`
* Almost Random: enter `ALMOST_RANDOM/bot.txt`
* MCTS: enter `MCTS/(r_)#_#_#.txt`
  * if there is an r, the simulation will be completely random
  * the first number is the number of simulations per iteration
  * the second number is the number of iterations per move
  * the third number is the number of games

After entering the name or file, you press enter again and can select the second player in the same way.

Now the game starts. If it is your turn, you can select a column by using arrow keys and pressing enter.

When the game is over, you will see a text stating who won. If you press any key, the game will start over but the player order will be reversed.

At any time, you can quit by clicking the x in the top right corner.

## ELO

I added a simple ELO system. You can see the ranking by running `./ranking.sh`.