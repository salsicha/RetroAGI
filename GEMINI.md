Update: 

The motor model should output sequences of tokens that are all the key presses to get to the short term goal (a transformer).
The whole system is a closed loop. At each iteration of the loop, the frontal and parietal models should not be recomputed. Instead the long and short goals should be saved. If they are achieved, they can be added to the training dataset.
The positive feedback loop closure is as follows: 
When long term goal achieved, stop and learn on that goal. 
When short term goal achieved, stop and learn that goal.
When temporal prediction is correct, stop and learn
When key sequence gets achieves the short term goal, stop and learn that sequence.
Negative feedback loop closure is as follows:
When death occurs, heavily negative weight long term goal, short term goal, and motor sequence, stop and lean to avoid those.
When temporal model fails to predict the sequence of motion for other sprites in the game, assign a negative value to that and stop and learn.
