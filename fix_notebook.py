import json

with open("Test14f1.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if "source" in cell:
        # replace lists of strings
        if isinstance(cell["source"], list):
            cell["source"] = [s.replace("/content/", "data/") for s in cell["source"]]
        elif isinstance(cell["source"], str):
            cell["source"] = cell["source"].replace("/content/", "data/")

with open("Test14f1.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Updated Test14f1.ipynb")
