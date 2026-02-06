

# TODO

The purpose of this project is to create a continuously learning agent with an architecture similiar to the human brain. The visual input from the Mario game is sent into the Occipital model. The Occipital output goees to a decoder that reconstructs the input image, and also send "what" latent parameters to the Temporal model and "where/how" latent parameters to the Parietal model. The Temporal model outputs to a decoder that semantically describes the sequence of events in the scene, and also to the Parietal model. The Parietal outputs to a decoder that outputs the areas of the scene that are Mario's short-term objectives/destinations, and also to the Motor, Frontal, and back to the Temporal model. The Frontal outputs to a decoder that outputs the long-term goals in the scene, and also back to the Parietal model. The Motor model outputs to a decoder that sends the next key press input back to the game.


1. [Done] Build Occipital model and decoder that read images from the game scene and reconstructs them with the decoder so that we know the Occipital model has learned the latent parameters of the visual field in the game.

2. Build Temporal model that takes latent inputs from the Occipital and Parietal models, and decoder that emit semantic sequences that describe the events in the games so that we know the Temporal model has learned the latent parameters of sequences of events in the game.

3. Build Parietal model that takes inputs from the Temporal model, Occipital model, and Frontal model, and decoder that outputs the areas of the scene that are Mario's short-term objectives.

4. Build Frontal model that takes inputs from the Parietal model, and decoder that outputs the long-term goals in the scene.

5. Build Motor model that takes in the latent output of the Parietal and uses a decoder to emit the next key press input to the game.


