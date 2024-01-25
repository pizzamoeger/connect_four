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
`./train_MCTS <random simulate (0/1)> <#simulations> <#iterations> <#games>`. 
The resulting MCTS will be stored in `data/MCTS/(r_)<#simulations>_<#iterations>_<#games>.txt`, with the `r_` being present if 
the simulate was random.

The `train_DQN` executable is for training the DQN. You can give optional command-line arguments. 
Each argument has to be given by a key and a value, which have to be seperated by a space. 
The keys are to be entered as followed:
* `c` for the Parameter c
* `games` for the amount of training games
* `batch_size` for the size of the batch that will be trained in each step
* `replay_buffer_size` to set the size of the replay memory
* `discount_factor` to set the discount factor
* `epsilon_red` to set how much epsilon will be reduced after each game
* `epsilon` to set the value of epsilon
* `fcw` to set the learning rate for the weights in a fully connected layer
* `fcb` to set the learning rate for the biases in a fully connected layer
* `cw` to set the learning rate for the weights in a convolutional layer
* `cb` to set the learning rate for the biases in a convolutional layer
* `fcwr` to set the reduction of learning rate for the weights in a fully connected layer
* `fcbr` to set the reduction of learning rate for the biases in a fully connected layer
* `cwr` to set the reduction of learning rate for the weights in a convolutional layer
* `cbr` to set the reduction of learning rate for the biases in a convolutional layer
* `L2` to set the L2 regularization
* `momcoef` to set the momentum coefficient

The architecture of the neural network has to be changed manually in `src/DQN/main.cpp`. 
For each type of hidden layer, there is an example provided in the code. The program will prompt 
you to enter the filename in which 
the DQN should be stored as soon as it is done with training. 
The relative path from the `data/DQN/` directory has to be entered. 

## Controls
If you ran the program without any arguments, a window will open. You can select the first player (manual or computer) by using the arrow key and then pressing enter. 
Now you will either be prompted for your name, which is pretty straight forward, or the file in which the artificial player is stored. For this you have the following options:
* Random: enter `RANDOM/bot.txt`
* Almost Random: enter `ALMOST_RANDOM/bot.txt`
* MCTS: enter `MCTS/(r_)<#simulations>_<#iterations>_<#games>.txt`
  * Note that if there is an r it means that the simulate of the MCTS is completely random.
* DQN: enter the relative path to your file from `data/`

After entering the name or file, you have press enter again and can select the second player in the same way.

Now the game starts. If it is your turn, you can select a column by using arrow keys and pressing enter.

When the game is over, you will see a text stating who won. If you press any key, the game will start over but the player order will be reversed.

At any time, you can quit by clicking the x in the top right corner.

## ELO

I added a simple ELO system. You can see the ranking by running `./ranking.sh`.
