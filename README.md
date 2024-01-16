# Connect Four

## Description
This is a C++ Program for playing Connect four. The UI is written in SDL2.
You can play against a friend or against the computer.
There are four different algorithms implemented against which can be played. 
One algorithm acts randomly and one plays optimal if in the next to moves the game can be won. 
The other two implemented algorithm are MCTS and DQN. 

## Requirements
This program uses SDL2 and SDL_ttf.

Install (Ubuntu): 

`$ sudo apt-get install libsdl2-dev`

`$ sudo apt-get install libsdl2-ttf-dev`

The neural network is implemented in CUDA, which requires a NVIDIA GPU.

## Compiling and running
To compile, you can simply run `cmake CmakeList.txt` and `make`.
It will produce three executables. 

The `connect_four` executable is for playing the game. It requires no command-line arguments. 

The `train_MCTS` executable is for training the MCTS. It requires four command-line arguments and should be run like this: 
`./train_MCTS <random simulation (0/1)> <#simulations> <#iterations> <#games>`. The resulting MCTS will be stored in `data/MCTS/(r_)<#simulations>_<#iterations>_<#games>.txt`, with the `r_` being present if 
the simulation was random.

The `train_DQN` executable is for training the DQN.
You will have to change the hyperparams manually in the code.
The results will be stored `data/DQN/(fc/conv)_<#batch_size>_<#c>_<#games>.txt`

## Controls
If you ran the program without any arguments, a window will open. You can select the first player (manual or computer) by using the arrow key and then pressing enter. 
Now you will either be prompted for your name, which is pretty straight forward, or the file in which the artificial player is stored. For this you have the following options:
* Random: enter `RANDOM/bot.txt`
* Almost Random: enter `ALMOST_RANDOM/bot.txt`
* MCTS: enter `MCTS/(r_)<#simulations>_<#iterations>_<#games>.txt`
  * Note that if there is an r it means that the simulation of the MCTS is completely random.
* DQN: enter `DQN/(fc/conv)_<#batch_size>_<#c>_<#games>.txt`

After entering the name or file, you have press enter again and can select the second player in the same way.

Now the game starts. If it is your turn, you can select a column by using arrow keys and pressing enter.

When the game is over, you will see a text stating who won. If you press any key, the game will start over but the player order will be reversed.

At any time, you can quit by clicking the x in the top right corner.

## ELO

I added a simple ELO system. You can see the ranking by running `./ranking.sh`.
