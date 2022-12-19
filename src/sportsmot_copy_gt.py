import os.path as osp
import os
import numpy as np
import shutil


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


# source_root = '/home/fc/datasets/sportsmot/images/val'
# dir_root = '/home/fc/PycharmProjects/TrackEval/data/gt/sportsmot/sportsmot-val'


# seqs = [s for s in os.listdir(source_root)]


# for seq in seqs:
#
#     shutil.copytree(osp.join(source_root, seq,'gt'), osp.join(dir_root, seq,'gt'))
#     shutil.copyfile(osp.join(source_root, seq, 'seqinfo.ini'), osp.join(dir_root, seq,'seqinfo.ini'))
# print(volleyball_count,football_count,basketball_count)


### check test results one-to-one match

source_root = '/home/fc/datasets/sportsmot/images/results/sportsmot_test_public_hrnet18_byte'
seqs = [s.split('.')[0] for s in os.listdir(source_root)]
seqs.sort()
print(seqs)
split_txt_root = '/home/fc/datasets/sportsmot/splits_txt'

test_txt = osp.join(split_txt_root, 'test.txt')
test_list = []
with open(test_txt, "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  # 去掉列表中每一个元素的换行符
        test_list.append(line)
test_list.sort()
print(test_list)
if seqs ==test_list:
    print('true')
else:
    print('false')
