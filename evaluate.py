import argparse
import numpy as np
import torch.nn as nn
import torch
from models.resnet import resnet50
from datasets import get_test_datasets
from utils.eval_utils import extract_features, get_ids, evaluate_core

market_dir = '/Users/zhulf/data/market'
reid_dir = '/Users/zhulf/data/reid_match/reid'

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=market_dir, help='Dataset directory')
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


def evaluate(model):
    print("="*20, "Extracting features from query and gallery ..." , "="*20)
    query_features = extract_features(model, test_dataloaders['query'])
    gallery_features = extract_features(model, test_dataloaders['gallery'])
    query_paths = test_image_datasets['query'].imgs
    gallery_paths = test_image_datasets['gallery'].imgs
    query_camera_ids, query_labels = get_ids(query_paths)
    gallery_camera_ids, gallery_labels = get_ids(gallery_paths)
    print("=" * 20, "Evaluating ...", "=" * 20)
    AP, CMC = 0, np.zeros(len(gallery_labels), dtype=np.float)
    for i in range(len(query_labels)):
        ap, cmc = evaluate_core(query_features[i], query_camera_ids[i], query_labels[i], gallery_features, gallery_camera_ids, gallery_labels)
        if cmc[0] == -1:
            continue
        CMC += cmc
        AP += ap
    CMC = CMC / len(query_labels)
    mAP = AP / len(query_labels)
    print("Rank@1:{}, Rank@5: {}, Rank@10: {}, mAP: {}".format(CMC[0], CMC[4], CMC[9], mAP))


if __name__ == '__main__':
    evaluate(model)