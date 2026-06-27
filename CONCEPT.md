
Hierarchical recursive, actor-worldmodel-critic, architecture for learning to play super mario bros. It is a progressive resolution machine, the resolution of the game and the model's knowledge of it increases as it "grows up". Learning starts with a very low resolution simulation of the Super Mario Bros game. There architecture for generating action is be a three layer heirarchy, a high level planner, low level planner, and an adaptive controller, but there is also LSTM world model that takes the grid and semantic information of the game and predicts the episodic outcome of the current state, and a heuristic critic model feeding criticism of the world model back into the actor model. The entire model is essentially a single neo-cortex column in the "thousands brain theory" of intelligence. The LSTM is basically the hippocampus.





