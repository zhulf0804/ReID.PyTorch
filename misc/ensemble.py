import os
import glob
import json
from collections import Counter


data_dir = '/Users/zhulf/data/reid_match/results'
json_paths = glob.glob(os.path.join(data_dir, '*.json'))

l = []
for json_path in json_paths:
    print(json_path)
    with open(json_path, 'r') as f:
        d = json.load(f)
    l.append(d)

keys = l[0].keys()

tmp = []
for i in range(len(l)):
    tmp.extend(l[i]['08002216.png'])
    print(l[i]['08002216.png'])

#count = Counter(tmp)
#print(count)
#print(len(count))
results = {}
for key in keys:
    d = {}
    for i in range(len(l)):
        for j in range(len(l[i][key])):
            if j < 5:
                d[l[i][key][j]] = d.get(l[i][key][j], 0) + (200 - j)
            elif j < 10:
                d[l[i][key][j]] = d.get(l[i][key][j], 0) + (200 - j) * 0.5
            elif j < 20:
                d[l[i][key][j]] = d.get(l[i][key][j], 0) + (200 - j) * 0.2
            elif j < 50:
                d[l[i][key][j]] = d.get(l[i][key][j], 0) + (200 - j) * 0.1
            elif j < 100:
                d[l[i][key][j]] = d.get(l[i][key][j], 0) + (200 - j) * 0.05
            else:
                d[l[i][key][j]] = d.get(l[i][key][j], 0) + (200 - j) * 0.025
    d = sorted(d.items(), key=lambda item: item[1], reverse=True)
    print(key, d)
    result = [t[0] for t in d]
    result = result[:200]
    results[key] = result

saved_file = 'ensemble.json'
with open(saved_file, 'w') as f:
    json.dump(results, f)

