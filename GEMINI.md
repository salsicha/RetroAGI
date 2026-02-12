
Update:

Create a tutor that bootstraps the learning process. Use the script tutor/segmentation/segment_inference.py as a starting place. The tutor should be able to identify sprite positions for the occipital model. Another tutor script should be able to identify near term goals, and another one for long term goals. The tutor can be used to teach the individual models the correct output, and it can also replace the output of the individual models when doing closed loop training so that dependent models have valid inputs. 

Also what is needed is a decoder for each model. Each model passes its latent parameters to the next model, but there also needs to be a decoder model that takes the latent output of a model and produces an output that can be verified. For the frontal and parietal models, the output is a hotspot on the screen that identifies the goal, for example.