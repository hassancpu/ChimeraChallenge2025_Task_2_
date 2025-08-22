import os
import json
import csv

# Path to your data directory
data_dir = "/scratch-shared/20215294/CHIM/data"
output_csv = "bladder-brs.csv"

rows = []
for case_id in sorted(os.listdir(data_dir)):
    case_path = os.path.join(data_dir, case_id)
    if not os.path.isdir(case_path):
        continue

    json_file = os.path.join(case_path, f"{case_id}_CD.json")
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            info = json.load(f)
            brs = info.get("BRS", None)
            if brs is None:
                print(f"Warning: BRS not found for {case_id}")
            else:
                slide_id = case_id + "_HE"
                rows.append([case_id, slide_id, brs])
    else:
        print(f"Warning: No JSON file found for {case_id}")

# Write the output CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case_id", "slide_id", "BRS"])
    writer.writerows(rows)

print(f"Saved {len(rows)} entries to {output_csv}")
