import ast
import pandas as pd

df = pd.read_csv("EPIC_100_train.csv")
cols = ["video_id", "start_frame", "stop_frame", "verb", "verb_class", "noun", "all_nouns", "all_noun_classes", "relevant"]

# Get an array indexed by noun/verb id. Mapping to relevance
verb_classes = pd.read_csv("classes/EPIC_verb_classes.csv", index_col="verb_id")["relevant"]
noun_classes = pd.read_csv("classes/EPIC_noun_classes.csv", index_col="noun_id")["relevant"]

# Given an annotation row, we mark it irrelevant if and only if:
    # verb_class is irrelevant
    # or
    # all nouns in all_noun_classes are irrelevant
def is_relevant(row):
    if not verb_classes[row["verb_class"]]:
        return False
    # Transform into a python array
    noun_ids = ast.literal_eval(row["all_noun_classes"])

    return any(noun_classes[id] for id in noun_ids)

df["relevant"] = df.apply(is_relevant, axis=1)

# result = df[df["video_id"] == "P01_01"][cols]
result = df[cols]
result.to_csv("annotations.csv", index=False)
print(f"Saved {len(result)} rows to annotations.csv")
