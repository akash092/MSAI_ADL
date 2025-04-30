import json
from pathlib import Path

with open("../data/valid_grader/balanced_qa_pairs.json") as f:
    validation = json.load(f)
validation_set = {(qa["question"], qa["answer"], qa["image_file"]) for qa in validation}
print(f"Loaded {len(validation_set)} validation samples.")

generated_files = Path("../data").glob("self_valid_qa_pairs.json")
generated_set = set()
total_generated = []
for file in generated_files:
    with open(file) as f:
        qa_pairs = json.load(f)
    #print(f"Loaded {len(qa_pairs)} samples from {file}.")
    #print(qa_pairs)
    generated_set.update((qa["question"], qa["answer"], qa["image_file"]) for qa in qa_pairs)
    total_generated.extend(qa_pairs)

matches = validation_set & generated_set
print(f"Matched {len(matches)} out of {len(validation_set)} unique QA pairs, percentage: {len(matches) / len(validation_set) * 100:.2f}%")
#print("Matches:", matches)
print("Unmatched:", validation_set - matches)
print("Total generated QA pairs:", len(total_generated))