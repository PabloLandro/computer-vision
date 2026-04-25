import pandas as pd

df = pd.read_csv("EPIC_100_train.csv")
result = df[df["video_id"] == "P01_01"]
result.to_csv("P01_01_annotations.csv", index=False)
print(f"Saved {len(result)} rows to P01_01_annotations.csv")
