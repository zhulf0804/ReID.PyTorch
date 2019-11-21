import os
import numpy as np
import torch


def fliplr(img):
    #flip horizontal
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def extract_features(model, dataloader):
    model = model.cuda()
    features = torch.FloatTensor()
    rt_labels, rt_camids = np.array([], dtype=np.int), np.array([], dtype=np.int)
    for data in dataloader:
        inputs, labels, camids = data
        rt_labels = np.append(rt_labels, labels)
        rt_camids = np.append(rt_camids, camids)
        n, _, _, _ = inputs.size()
        outputs_fuse = torch.zeros(n, 512, dtype=torch.float)
        for i in range(2):
            if i == 1:
                inputs = fliplr(inputs)
            inputs = inputs.cuda()
            outputs, _ = model(inputs)
            outputs, _ = model(inputs)
            outputs_fuse += outputs.cpu()
            inputs = inputs.cpu()

        # l2 norm
        outputs_fuse_norm = torch.norm(outputs_fuse, p=2, dim=1, keepdim=True)
        outputs_fuse = outputs_fuse / outputs_fuse_norm
        features = torch.cat([features, outputs_fuse], dim=0)
    return features.numpy(), rt_labels, rt_camids


def compute_mAP(inds, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(inds))
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(inds, junk_index, invert=True)
    inds = inds[mask]

    # find good index
    mask = np.in1d(inds, good_index)
    rows_good = np.argwhere(mask == True).squeeze(1)
    #print(rows_good)
    cmc[rows_good[0]:] = 1
    ngood = len(good_index)
    for i in range(ngood):
        precision = (i + 1) / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i / rows_good[i]
        else:
            old_precision = 1.0
        ap += ((old_precision + precision) / 2) / ngood
    return ap, cmc


def evaluate_core(qf, qc, ql, gfs, gcs, gls):
    qf = np.reshape(qf, (-1, 1))
    scores = np.dot(gfs, qf)
    scores = np.squeeze(scores, axis=1)
    inds = np.argsort(scores)[::-1]
    query_index = np.argwhere(gls == ql).squeeze(1)
    camera_index = np.argwhere(gcs == qc).squeeze(1)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gls == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index1, junk_index2)
    return compute_mAP(inds, good_index, junk_index)



def get_filenames(img_path):
    filenames = []
    for path, v in img_path:
        filename = os.path.basename(path).strip()
        filenames.append(filename)
    return filenames


def inference(qf, gfs, top=200):
    qf = np.reshape(qf, (-1, 1))
    scores = np.dot(gfs, qf)
    scores = np.squeeze(scores, axis=1)
    inds = np.argsort(scores)[::-1]
    inds = inds[0:top]
    return inds