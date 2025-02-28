import json
import sys
from pathlib import Path

# <<< encoding UTF8 !!!!! >>>
# print(sys.argv)

j = None
if len(sys.argv[1]) < 2:
    print("No file provided")
    exit()

path = Path(sys.argv[1])
with open(path, "r", encoding="utf-8") as f:
    j = json.load(f)

destination = path

# print(len())
cells = j["cells"]
deleted = 0
cells = j["cells"]
deleted = 0
for i in range(len(cells) - 1, -1, -1):
    cell = cells[i]
    if cell["cell_type"] == "code":
        if len(cell["outputs"]) > 0:
            to_remove = []
            for i, output in enumerate(cell["outputs"]):
                if "name" in output and output["name"] == "stderr":
                    to_remove.append(i)
            to_remove = reversed(to_remove)
            for i in to_remove:
                cell["outputs"].pop(i)
                deleted += 1
# print(f"deleted {deleted} cells")
destination = path  # path.with_name(path.stem + "_stripped.ipynb")
# print(f"deleted {deleted} cells")
# destination = path.with_name(path.stem + "_stripped.ipynb")
# destination = path.with_stem(path.stem + "_stripped")
# print(f"writing to {destination}")
if deleted > 0:
    with open(destination, "w", encoding="utf-8") as f:
        json.dump(j, f, indent=1)
        f.write("\n")

print(json.dumps(j, indent=1))
