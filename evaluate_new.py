import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from utils.eval_utils import extract_features, evaluate_core
from config import cfg
from models.resnet import resnet50
from data.build import make_dataloader

CELoss = CrossEntropyLoss()
def evaluate(model, query_loader, test_loader):
    print("="*20, "Extracting features from query and gallery and evaluating" , "="*20)
    model.eval()

    with torch.no_grad():
        query_features, query_labels, query_camera_ids = extract_features(model, query_loader)
        gallery_features, gallery_labels, gallery_camera_ids = extract_features(model, test_loader)
    print(query_labels)
    print(gallery_labels)
    AP, CMC = 0, np.zeros(len(gallery_labels), dtype=np.float)
    for i in range(len(query_labels)):
        ap, cmc = evaluate_core(query_features[i], query_camera_ids[i], query_labels[i], gallery_features, gallery_camera_ids, gallery_labels)
        if cmc[0] == -1:
            continue
        CMC += cmc
        AP += ap
    CMC = CMC / len(query_labels)
    mAP = AP / len(query_labels)
    return CMC, mAP
    #print("Rank@1:{}, Rank@5: {}, Rank@10: {}, mAP: {}".format(CMC[0], CMC[4], CMC[9], mAP))

if __name__ == '__main__':
    train_loader, test_loader, query_loader = make_dataloader(cfg)
    model = resnet50(cfg.DATASET.NUM_CLASS)
    model.load_state_dict(torch.load('checkpoints/resnet50_60.pth'))
    model.classifier.classifier = nn.Sequential()
    print(evaluate(model, query_loader, test_loader))