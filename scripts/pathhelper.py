from pathlib import Path

root = Path(".")
prefix = "cmuh_"

for path in root.rglob("*.json"):
    if path.name.startswith(prefix):
        #remove _ from "cmuh_", i.e. "cmuh_1234.json" -> "cmuh1234.json"
        new_name = path.name.replace("_", "", 1)
        new_path = path.with_name(new_name)
    else:
        new_path = path.with_name(prefix + path.name)
    if new_path.exists():
        print(f"skipping, already exists:{new_path}")
        continue
    path.rename(new_path)
    print(f"Renamed: {path} -> {new_path}")

