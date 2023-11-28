# Connect Four

## Description
This is a C++ Program for playing Connect four. It is written in SDL2.
You can play against a friend or against the computer.

## Requirements
This program uses SDL2, SDL_ttf and boost multiprecision.

Install (Ubuntu): 
sudo apt-get install libsdl2-dev
sudo apt-get install libsdl2-ttf-dev
sudo apt install libboost-all-dev

## Compiling and running
To compile, you can simply run `cmake CmakeList.txt` and then `make`.
To run, you can run `./connect_four`.
You can also train an MCTS. To do so, run `./connect_four <random sim (1/0)> <#sims> <#iterations> <#games>`.
The results will be stored in the file `MCTS/(r_)#sims_#iterations_#games.txt`.

## Controls
If you ran the program without any arguments, a window will open. You can select the first player (manual or computer) by using the arrow key and then pressing enter. Now you will either be prompted for your name, which is pretty straight forward, or the file in which your opponent is stored. For this you have the following options:
* Random: enter `RANDOM/bot.txt`
* Almost Random: enter `ALMOST_RANDOM/bot.txt`
* MCTS: enter `MCTS/(r_)#sims_#iterations_#games.txt`
  * Note that if there is an r it means that the simulation of the MCTS is completely random.

After entering the name or file, you press enter again and can select the second player in the same way.

Now the game starts. If it is your turn, you can select a column by using arrow keys and pressing enter.

When the game is over, you will see a text stating who won. If you press any key, the game will start over but the player order will be reversed.

At any time, you can quit by clicking the x in the top right corner.

## ELO

I added a simple ELO system. You can see the ranking by running `./ranking.sh`.