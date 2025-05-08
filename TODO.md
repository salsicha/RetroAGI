


# TODO

1. Train segment model  
 - DONE  

2. Run segment inference  
 - run from cell in mario notebook  
 - DONE  

3. Mario elevation  
 - work out method for getting Mario's distance from bottom of game screen   
 - DONE  
 
4. Create progressive perceptive field and occupancy detection for agent  
 - Segment game screen into progressive grid  
 - Create method for detecting the grid position of each sprite  
 - DONE  

5. Create behavior tree for apply user generated constraints  
 - DONE

6. Create tokens and transformer for actions  
 - DONE

7. Create method for discovering motifs
 - feed mario position into stumpy for action clustering  
 - DONE

8. Build system for training new DNNs on sequence clusters  

9. Build system for generating new tokens and retraining transformer  

10. Evaluation method for deterritorializing bad tokens  



Objective:  

The obective is MPC, which the predictions are single tokens that represent long strings of atomic actions  
The transformer produces a tokens (motifs) that represents a string of actions  
We look at what the predicted outcome and assign a reward  
We look at the top tokens in the transformer outout and evaluate each of them  
We pick the top performing token  
We constantly retrain the transfrmer on successful runs  
But we also look for progressively longer motifs to create new tokens  
The reward function is a field around boxes and coins and away from enemies  
And a global reward leading to the end of the level  
We then have to figure out how to forget old/unused tokens  
Finally, the behavior tree functions as a regularization and human in the loop feedback  



Notes:  
Add dimension for time, all negative values, most recent is -1  
Embeddings are just unit vectors  
Clusters are just sums of action vectors  
Action tokens and entity tokens  
Entity tokens use same space vectors as action and groups  
But also have dimensions for entity types  


Token ID Action Embedding (vector)  
1, "up",     [0.0, 1.0]  
2, "down",   [0.0, -1.0]  
3, "left",   [-1.0, 0.0]  
4, "right",  [1.0, 0.0]  
5, "group1", [2.0, 0.0] # This could be two right actions chained together as an example  
....  


Reward Functions:  
First order magnitude: get to end of level  
Second order magnitude: get points  



DQN
https://github.com/vpulab/Semantic-Segmentation-Boost-Reinforcement-Learning

