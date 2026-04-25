import pandas as pd

df = pd.read_csv("EPIC_100_train_clean.csv")
cols = ["video_id", "start_frame", "stop_frame", "verb", "verb_class", "noun", "all_nouns", "all_noun_classes"]
result = df[df["video_id"] == "P01_01"][cols]

result.to_csv("P01_01_annotations.csv", index=False)
print(f"Saved {len(result)} rows to P01_01_annotations.csv")
