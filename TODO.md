

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

7. Implement DQN for filtering action tokens???  

8. Create method for discovering motifs
 - get vertical position of mario on screen, feed into matrixprofile method for action classification (STAMP, STOMP, SCRIMP+, ???), see if forward/backward move commands can be added as another dimension  

9. Build system for training new DNNs on sequence clusters  

10. Build system for generating new tokens and retraining transformer  

11. Evaluation method for deterritorializing bad tokens  




Notes:  
Make starting embedding table  
Mario height calculation in Mario notebook  
Make perceptive field  
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



