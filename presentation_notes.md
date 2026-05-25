We will probably be asked to explain what we did and then they will make some specific questions about the models

# Our presentation

## Goal of the project
Our first goal was to use V-Jepa2 to label actions in a video and elaborate a text description of the video, with the help of an LLM to transform it into a cohesive text.
We quickly realized that this being so general is not feasible, as V-Jepa only produces embeddings for blocks of frames and it relies on us training a classifier over those embeddings. We would need to train a very complex action classifier. That is why we decided to limit the scope to a very specific type of videos. This is why we came to the decision of using cooking videos.

We found a dataset of egocentric POV cooking videos, which is a great fit for our use case.

Our goal is to produce a detailed list of cooking actions happening in a video, which we can feed into an LLM to produce a recipe.

## Challenges
### Noise of the data.
A great part of the video is not relevant to our goal, actions such as washing dishes or grabbing utensils from a cupboard should be ignored.

Our dataset contains labels indicating what action is being performed by the cook at all times, this actions are labeled using a set of 900 classes. We chose ~500 of this actions which we consider relevant.

This not only allowed us to filter and train on cleaner data, but also to train a relevance classifier to filter a video when in inference time.

### Small window
Our V-Jepa model takes as input 64 frames, which given the framerate of our videos, is about 2 seconds of video. But most of our actions are longer than 2 seconds, so to have a better classification we used an encoder that takes as input 7 embeddings, this gives us 15 seconds of context, which provides more robust data.

## Pipeline
Our model works the following way:
1. Preprocess the video by cropping to the center of the content.
2. Divide the video in a sequence of 64 frame blocks.
3. Run V-Jepa on each block to obtain an array of embeddings.
4. Feed a 7 block window of embeddings to an encoder transformer layer: This allows the current block embedding to be aware of the surrounding embeddings.
5. MLP heads:
   - Relevance filtering
   - Action classification
6. Event sequence fed to an LLM

# Other considerations
## Reason for using a transformer
We also tried using a simple MLP over the encoding, but this produced only 70% accuracy in the action labeling.

## Why 

## Points of improvement (just give some ideas)

We could benefit of a model that natively supports bigger blocks of frames.

A separate classifier may prove useful to precisely classify ingredients with high granularity.

# Project Demo
I would only show two things:
- (Maybe) structure of a embedding pkl to show how we structure the outputs of V-Jepa
- Event list produced by an unseen video
- Recipe produced by an unseen video
- The unseen video used (important to have it already downloaded, online it's very laggy)

# Possible questions

- Input and output of each step (should prepare a small cheatsheet with exact dimensions and so on)
- What we used to implement it:
  - Models in pytorch:
    - Encoder: `TransformerEncoderLayer`
    - MLP: `Linear -> GELU -> Dropout -> Linear`