This is not a "standard" or "off-the-shelf" architecture in the way that standard ResNets, BERT, or GPT models are. However, it is a very specific implementation of a well-known family of research architectures.

It combines two major concepts from advanced deep learning research: Hierarchical/Multi-Timescale Modeling and Discrete Latent Bottlenecks.

Here is a breakdown of how common this is and how effective it is in practice.

Is it common?
The exact combination you have built (two causal transformers at different timescales, bridged purely by an upsampled Gumbel-Softmax embedding addition, with zero auxiliary loss on the higher level) is quite specialized. However, its foundational concepts show up frequently in cutting-edge research:

Hierarchical Reinforcement Learning (HRL): Architectures like FeUdal Networks use a "Manager-Worker" dynamic almost exactly like this. A Manager operates at a slow timescale to output "goals" (Stream A), and a Worker operates at a fast timescale to output "actions" to achieve those goals (Stream B).
Clockwork Architectures: Models like Clockwork RNNs or modern Hierarchical Transformers deliberately process data at different frequencies to handle very long sequences efficiently without running out of memory.
Discrete Latent Variables: Using Gumbel-Softmax (or Vector Quantization, as seen in VQ-VAE) to force a model to pass a strict, discrete "concept" or "token" to another network is a highly common technique in representation learning and audio/video generation.
How effective is it?
In theory, this architecture is extremely powerful. In practice, it is notoriously difficult to train and stabilize.

Where it excels (The Good):

Unsupervised Concept Discovery: Because Stream A is not given explicit targets, it is forced to invent its own "language" to communicate with Stream B. It can discover useful abstractions or macro-actions that humans might not have even thought to label.
Long-Context Efficiency: If you want an AI to play Super Mario Bros for 5 minutes, a standard Transformer would choke on the millions of frames. This architecture allows Stream A to think in terms of "rooms" or "seconds," vastly expanding the model's memory horizon while Stream B handles the frame-by-frame jumping.
Where it struggles (The Bad):

The "Cold Start" and Credit Assignment: At step 0, Stream A sends random garbage to Stream B. Stream B tries to make sense of the garbage, fails, and sends a gradient back. This gradient basically tells Stream A, "Whatever you sent me didn't help." It can take a very long time for the models to accidentally stumble upon a useful token communication. This often leads to models getting stuck in local minima early on.
Mode Collapse: Often, Stream A will realize that changing its token confuses Stream B, so Stream A will just learn to output the exact same token (e.g., Token #4) every single time. Stream B then just learns to ignore Stream A entirely.
Gumbel-Softmax Instability: The Straight-Through Estimator (using hard=True) is computationally "biased." The gradient that flows backwards is an approximation. Over complex tasks, this approximation can degrade, causing learning to stall.
How Researchers Make it More Effective
If you were to scale this up for your RetroAGI project, researchers typically add a few mechanisms to make this architecture highly effective:

Cross-Attention Instead of Addition: Instead of naively adding pred_emb_A_upsampled to Stream B's embeddings, Stream B would use a Cross-Attention layer to actively "look" at Stream A's output. This makes it easier for Stream B to incorporate the information.
Auxiliary Losses: To prevent the "Cold Start" problem, researchers often give Stream A a tiny bit of direct supervision (e.g., an auxiliary loss to predict the overall task reward or a coarse target) just to point its weights in the right direction before letting the implicit gradients take over.
Temperature Annealing: The tau (temperature) parameter in F.gumbel_softmax is usually started high (e.g., tau=5.0, which makes it very continuous and easy to train) and slowly decayed to tau=0.1 (making it strictly discrete) over the course of training.

Next steps?
Use scriptable scenarios with specific goals for training.
Currently the heirarchy is only two layers, the bottom layer would ideally be an adaptive controller
Is there a trained critic model available online?
Integrate the trained vision model?
Could predictive coding between the first and second layer help online training?
