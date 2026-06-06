
Hierarchical recursive, actor-worldmodel-critic, architecture for learning to play super mario bros. It is a progressive resolution machine, the resolution of the game and the model's knowledge of it increases as it "grows up". Learning starts with a very low resolution simulation of the Super Mario Bros game. There architecture for generating action is be a three layer heirarchy, a high level planner, low level planner, and an adaptive controller, but there is also be world model that takes the predicted actions and predicts the outcome of the action, and a critic model feeding criticism of the world model back into the actor model. The entire model is essentially a single neo-cortex column in the "thousands brain theory" of intelligence. Except the LSTM, which is basically the hippocampus.

Next steps:

1. Build a vision transformer model that extracts position and semantic information from the simplified pygame version of SMB. Align its interfaces with the "vision" model that works on the linear version of the hierarchical model and the segmentation model that has already been trained on the full version of SMB.

2. Separate hierarchical learning architecture into different levels of resolution. Low level works on linear sequenceses. Next level works on simplified pygame version of SMB. Top level works on the full version of SMB.


Old notes:
Make adaptive controller an adaptive model predictive controller (AMPC)? The World Model is the LSTM. Does the LSTM operate in the adaptive controller? No, too slow. Loop the LSTM with A or B transformer or both?
Critic model is made of heuristics.
Split engine into high and low resolution scripts with compatible interfaces.
LSTM preditive/memory model could include grid cells for place on the screen in high resolution version.
Break models into separate files with tests.
You could have grid cells since the decoding is super simple, they already make a 2d array of the input!
Wait! Can everything be grid activations??? Video? Sounds? Language? All ideas are grid spaces. Grid space creation is learned. Learning and ideation is the bridging of grid spaces!!!
This is why all knowledge is embedded!!!
Grid cells are hexagonal.
Invisible box problem: Need "place" neurons, not just predicted actions. An area needs a "qualia". Grid cells make a place, and the other senses turn a place into an "episode". Mario needs to remember places in the game. One network should be able to learn lots of places. It’s not just "where", it’s also "what".
How is "what" constructed? Communicated? Grids can have grids, each grid is learned, but then another grid can connect those grids. LSTM... 
RetroAGI is a single column in the thousands brain architecture 
The neo cortex is 6 layers, LLMs are like 30???

Create a vision transformer for reading the pygame video frames, make the labels and tokens match the full version's labels.
The vision transformer outputs to the actor model level A transformer.
Instead of training the critic, use the pygame score as the critic.
Integrate the existing trained vision model for visually understanding video frames from the full super mario bros game.
The adaptive controller should control how much each of the 4 direction keys are pressed. 0 or 1 for each key.
Break the scripts into separate low-res and high-res versions, where the high-res version starts with the trained weights of the low res version but then continues to learn.
Run in the full mario game.




