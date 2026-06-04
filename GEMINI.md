
Hierarchical recursive, actor-worldmodel-critic, architecture for learning to play super mario bros. There architecture for generating action is be a three layer heirarchy, a high level planner, low level planner, and an adaptive controller, but there is also be world model that takes the predicted actions and predicts the outcome of the action, and a critic model feeding criticism of the world model back into the actor model.

Next steps:

Add badguys to the platforms in the pygame.
Create a vision transformer for reading the pygame video frames, make the labels and tokens match the full version's labels.
The vision transformer outputs to the actor model level A transformer.
Instead of training the critic, use the pygame score as the critic.
Integrate the existing trained vision model for visually understanding video frames from the full super mario bros game.
The adaptive controller should control how much each of the 4 direction keys are pressed. 0 or 1 for each key.
Break the scripts into separate low-res and high-res versions, where the high-res version starts with the trained weights of the low res version but then continues to learn.
Run in the full mario game.
