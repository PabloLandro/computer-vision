import ast
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

ANNOTATIONS_FILE = REPO_ROOT / "data" / "annotations" / "annotations_train_test.csv"
CLASSES_DIR = REPO_ROOT / "data" / "classes"


def read_annotations() -> list[dict[str, str]]:
    with ANNOTATIONS_FILE.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def parse_nouns(value: str) -> list[str]:
    nouns = ast.literal_eval(value)
    if not isinstance(nouns, list):
        raise ValueError(f"Expected all_nouns to contain a list, got: {value}")
    return nouns


def write_class_file(class_keys: list[str], id_column: str, output_name: str) -> dict[str, int]:
    output_path = CLASSES_DIR / output_name
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[id_column, "class_key"])
        writer.writeheader()
        for class_id, class_key in enumerate(class_keys):
            writer.writerow({id_column: class_id, "class_key": class_key})

    print(f"Saved {len(class_keys)} rows to {output_path}")
    return {class_key: class_id for class_id, class_key in enumerate(class_keys)}


def write_updated_annotations(
    rows: list[dict[str, str]], verb_ids: dict[str, int], noun_ids: dict[str, int]
) -> None:
    fieldnames = list(rows[0].keys())

    for row in rows:
        row["verb_class"] = str(verb_ids[row["verb"]])
        row["all_noun_classes"] = str([noun_ids[noun] for noun in parse_nouns(row["all_nouns"])])

    with ANNOTATIONS_FILE.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {ANNOTATIONS_FILE}")


def main() -> None:
    CLASSES_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_annotations()
    if not rows:
        raise ValueError("No annotation rows found")

    verb_class_keys = sorted({row["verb"] for row in rows})
    noun_class_keys = sorted({noun for row in rows for noun in parse_nouns(row["all_nouns"])})

    verb_ids = write_class_file(verb_class_keys, "verb_id", "EPIC_verb_classes.csv")
    noun_ids = write_class_file(noun_class_keys, "noun_id", "EPIC_noun_classes.csv")
    write_updated_annotations(rows, verb_ids, noun_ids)


if __name__ == "__main__":
    main()
