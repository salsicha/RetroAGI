

# TODO

1. Train segment model
 - done

2. Run segment inference
 - add segment inference to scripts
 - run from cell in mario notebook

3. Mario elevation
 - work out method for getting Mario's distance from bottom of game screen 

4. Create progressive perceptive field and occupancy detection for agent
 - Segment game screen into progressive grid
 - Create method for detecting the grid position of each sprite

5. Create behavior tree for apply user generated constraints

6. Create tokens and transformer for actions

7. Implement DQN for filtering action tokens

8. Create method for discovering motifs
 - get vertical position of mario on screen, feed into matrixprofile method for action classification (STAMP, STOMP, SCRIMP+, ???), see if forward/backward move commands can be added as another dimension

9. Build system for training new DNNs on sequence clusters

10. Build system for generating new tokens and retraining transformer

11. Evaluation method for deterritorializing bad tokens


