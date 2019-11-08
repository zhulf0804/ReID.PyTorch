import json
from collections import OrderedDict


result_file = '/Users/zhulf/data/reid_match/reid/first_result.json'
example_file = '/Users/zhulf/data/reid_match/raw/submission_example_A.json'
saved_file = '/Users/zhulf/data/reid_match/reid/first_result_2.json'

with open(result_file, 'r') as f:
    results = json.loads(f.read())
with open(example_file, 'r') as f:
    examples = json.loads(f.read())

result_keys = list(results.keys())
example_keys = list(examples.keys())

#for i in range(1348):
#    if result_keys[i].strip() != example_keys[i].strip():
#        print(i, result_keys[i], example_keys[i])
d = {}
for key in example_keys:
    key = key.strip()
    result = [item.strip() for item in results[key]]
    d[key] = result

with open(saved_file, 'w') as f:
    json.dump(d, f)



