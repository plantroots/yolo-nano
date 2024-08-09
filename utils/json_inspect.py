import json

from collections import Counter


def summarize_structure(data, counter=None):
    if counter is None:
        counter = Counter()

    if isinstance(data, dict):
        for key, value in data.items():
            counter[key] += 1
            summarize_structure(value, counter)
    elif isinstance(data, list):
        for item in data:
            summarize_structure(item, counter)

    return counter


# with open("/home/plantroot/Datasets/COMPARE/original/instances_val2017.json", "r") as f:
#     data = json.load(f)

with open("/home/plantroot/Datasets/COMPARE/mine/instances_val2017.json", "r") as f:
    data = json.load(f)


summary = summarize_structure(data)
print(summary.most_common())
