# About

This is an attempt at creating a  deep reinforcement learning AI for the board game Connect Four.

## Model
The model takes inspiration from DeepMind's Atari model, but with a few tweaks. First of all, only a single Conv2D with a 4x4 kernel is used. In theory, each of the resulting (3 x 4 x N_DIM) should represent all of the most important information about the state of the game. This is then flattened and sent to a wide dense layer. The final layer is a Dense with 7 dimensions, one for each potential move.

## Training
 
 Training is done in AI vs AI mode:
 ```python connect_four.py ava --epsilon=1.```
 
 You can play the AI in Human vs AI mode:
 ```python connect_four.py hva```
