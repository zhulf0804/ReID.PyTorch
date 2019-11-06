import json
import argparse
import torch.nn as nn
import torch
from models.resnet import resnet50
from datasets import get_test_datasets
from utils.eval_utils import extract_features, get_filenames, inference

reid_dir = '/Users/zhulf/data/reid_match/reid'
example_file = '/Users/zhulf/data/reid_match/raw/submission_example_A.json'

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=reid_dir, help='Dataset directory')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_classes', type=int, default=751, help='Batch size')
parser.add_argument('--checkpoint', type=str, default='', help='Directory to save checkpoints')
args = parser.parse_args()


test_image_datasets, test_dataloaders, dataset_sizes, class_names = get_test_datasets(args.data_dir, args.batch_size)
num_classes = len(class_names)
model = resnet50(num_classes=args.num_classes)
model.load_state_dict(torch.load(args.checkpoint))
model.classifier.classifier = nn.Sequential()
print(model)


def predict(model):
    print("=" * 20, "Extracting features from query and gallery ...", "=" * 20)
    model.eval()
    with torch.no_grad():
        query_features = extract_features(model, test_dataloaders['query'])
        gallery_features = extract_features(model, test_dataloaders['gallery'])
    query_paths = test_image_datasets['query'].imgs
    gallery_paths = test_image_datasets['gallery'].imgs
    query_filenames = get_filenames(query_paths)
    gallery_filenames = get_filenames(gallery_paths)

    print("=" * 20, "Predicting ...", "=" * 20)
    saved_json = '/root/data/reid/first_result.json'
    d = {}
    for i in range(len(query_filenames)):
        index = inference(query_features[i], gallery_features, top=200)
        d[query_filenames[i].strip()] = gallery_filenames[index].tolist()
    
    with open(example_file, 'r') as f:
        examples = json.loads(f.read())
    example_keys = list(examples.keys())
    results = {}
    for key in example_keys:
        key = key.strip()
        result = [item.strip() for item in d[key]]
        results[key] = result
    with open(saved_json, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    predict(model)