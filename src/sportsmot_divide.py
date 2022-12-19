import os.path as osp
import os
import numpy as np
import shutil

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


# seq_root = '/data/yfzhang/MOT/JDE/MOT15/images/train'
# label_root = '/data/yfzhang/MOT/JDE/MOT15/labels_with_ids/train'
seq_root = '/home/fc/datasets/sportsmot/images/train'
label_root = '/home/fc/datasets/sportsmot/labels_with_ids/train'
split_txt_root = '/home/fc/datasets/sportsmot/splits_txt'
out_root = '/home/fc/datasets/sportsmot_divide/sportsmot_30'
out15_root = '/home/fc/datasets/sportsmot_divide/sportsmot_15'

seqs = [s for s in os.listdir(seq_root)]
# seqs = ['ADL-Rundle-6', 'ETH-Bahnhof', 'KITTI-13', 'PETS09-S2L1', 'TUD-Stadtmitte', 'ADL-Rundle-8', 'KITTI-17',
#         'ETH-Pedcross2', 'ETH-Sunnyday', 'TUD-Campus', 'Venice-2']


train_txt = osp.join(split_txt_root, 'train.txt')
train_list=[]
with open(train_txt, "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        train_list.append(line)
        # print(line)
# train_list = np.loadtxt(train_txt, dtype=np.float64)

volleyball_txt = osp.join(split_txt_root, 'volleyball.txt')
volleyball_list=[]
with open(volleyball_txt, "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        volleyball_list.append(line)




basketball_txt = osp.join(split_txt_root, 'basketball.txt')
basketball_list=[]
with open(basketball_txt, "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        basketball_list.append(line)


football_txt = osp.join(split_txt_root, 'football.txt')
football_list=[]
with open(football_txt, "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        football_list.append(line)


volleyball_count=0
football_count=0
basketball_count=0

for seq in seqs:
    if os.path.exists(osp.join(out_root,seq)):
        continue
    shutil.copytree(osp.join(seq_root, seq), osp.join(out15_root, seq))
    # if seq in volleyball_list and volleyball_count<10 :
    #     shutil.copytree(osp.join(seq_root,seq),osp.join(out_root,seq))
    #     volleyball_count+=1
    # elif seq in football_list and football_count<10 :
    #     shutil.copytree(osp.join(seq_root,seq),osp.join(out_root,seq))
    #     football_count += 1
    # elif seq in basketball_list and basketball_count<10 :
    #     shutil.copytree(osp.join(seq_root,seq),osp.join(out_root,seq))
    #     basketball_count+=1
    # else:
    #     print('something wrong')


# print(volleyball_count,football_count,basketball_count)


