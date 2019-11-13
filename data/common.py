import os.path as osp


def parse_market1501(path):
    img_path = path
    basename = osp.basename(path)
    pid = int(basename.split('_')[0])
    camid = int(basename.split('c')[1][0])
    return img_path, pid, camid


if __name__ == '__main__':
    print(parse_market1501('/Users/zhulf/Downloads/Market-1501-v15.09.15/bounding_box_test/-1_c1s1_000401_03.jpg'))
    print(parse_market1501('/Users/zhulf/Downloads/Market-1501-v15.09.15/bounding_box_test/1501_c6s4_001902_01.jpg'))

