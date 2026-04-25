We use this script to generate the embeddings for a whole video. This model of VJEPA takes blocks of 64 frames to generate a single embedding.

Our strategy is to split the video into a sequence of 64 frame blocks and then obtain an array of this embeddings, which we can then use for classification.