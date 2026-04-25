# Video

We download the video using annotation from the `video_downloader` folder and crop it to fit to the model's dimensions. For `vjepa2-vitl-fpc64-256` we use 256x256 pixels.

# Annotation

We take the original embeddings from [EPIC_100_train.csv](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/master/EPIC_100_train.csv).

We then use the clean script to remove unnecessary columns, the important ones we end up with are the `video_id`, `frame_start` and `frame_stop` of each action and the verbs and nouns for each action.