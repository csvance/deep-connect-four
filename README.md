# About

This is an attempt at creating a reinforcement learning "Connect Four" AI. Because Connect Four only has 144 (6 rows x 8 columns x 3) states and at most 8 potential actions, in theory we should be able to learn what action to take in each of these discrete states.

## Training
 
 Training is done in AI vs AI mode:
 ```python connect_four.py ava --epsilon 1.0 --decay 0.99999 ```
