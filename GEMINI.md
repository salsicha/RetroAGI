
Hierarchical recursive, actor-worldmodel-critic, architecture for learning to play super mario bros. It is a progressive resolution machine, the resolution of the game and the model's knowledge of it increases as it "grows up". There architecture for generating action is be a three layer heirarchy, a high level planner, low level planner, and an adaptive controller, but there is also be world model that takes the predicted actions and predicts the outcome of the action, and a critic model feeding criticism of the world model back into the actor model. The entire model is essentially a single neo-cortex column in the "thousands brain theory" of intelligence. Except the LSTM, which is basically the hippocampus.

Next steps:

Add badguys to the platforms in the pygame.
Add enemies and all other mechanics to low res pygame version.
Make adaptive controller an adaptive model predictive controller (AMPC)
Predictive controller can start off as heuristic, but must become LSTM in high resolution version.
LSTM preditive/memory model should include grid cells for place on the screen. 
Split engine into high and low resolution scripts with compatible interfaces
Break models into separate files with tests
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




