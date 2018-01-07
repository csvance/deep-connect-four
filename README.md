# About

This is an attempt at creating a  deep reinforcement learning AI for the board game Connect Four.

## Model
The model takes inspiration from DeepMind's Atari model, but with a few tweaks. First of all, only a single Conv2D with a 4*4 kernel is used. In theory, each of the resulting 3*4*N_DIM should represent all of the most important information about the state of the game. This is then flattened and sent to a dense layer.

## Training
 
 Training is done in AI vs AI mode:
 ```python connect_four.py ava --epsilon=1.```
 
 You can play the AI in Human vs AI mode:
 ```python connect_four.py hva```
