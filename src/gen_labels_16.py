import os.path as osp
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image,ImageDraw

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


######################### id label test##################################################
# iamge_path = '/home/fc/datasets/MOT17/train/MOT17-05-FRCNN/img1/000010.jpg'
# label_path = '/home/fc/datasets/MOT17Labels/labels_with_ids/train/MOT17-05-FRCNN/img1/000010.txt'
#
# img=Image.open(iamge_path)#显示图像
# img_original=Image.open(iamge_path)#显示图像
# img_original.show()
# print(img.size) # 图像尺寸
# width,height=img.size
# label = np.loadtxt(label_path, dtype=np.float64, delimiter=' ')
# draw = ImageDraw.Draw(img)
# count=0
# for _, tid, x, y, w, h in label:
#     count+=1
#     # x += w / 2
#     # y += h / 2
#     # label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
#     #     #             tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
#     object_width=w*width
#     object_height =h*height
#     x1= x*width - object_width/2
#     y1= y*height - object_height/2
#     x2=x*width +object_width/2
#     y2=y*height + object_height/2
#     if  tid==4 or tid==14 :
#         draw.rectangle([x1, y1, x2, y2], outline=(count*10, count*10, count*10), width=2)
#         text = 'id {:.6f}'.format(tid)
#         draw.text([x1, y1], text, (count*5, count*5, count*5))
#
# img.show() #显示
# print(count)
######################### id label test##################################################

######################### id label generate##################################################
seq_root = '/home/fc/datasets/MOT17/images/train'
label_root = '/home/fc/datasets/MOT17/labels_with_ids/train'


mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]
print(seqs)

tid_curr = 0  # 当前目标ID
tid_last = -1
for seq in seqs:
    mkdirs(osp.join(seq_root, seq, "imgs"))
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

    for fid, tid, x, y, w, h, mark, label, _ in gt:
        if mark == 0 or not label == 1:  # mark==0 意味着目标不需要被考虑，label not ==1 意味着 不是行人，不需要考虑
            continue
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        # [class] [identity] [x_center] [y_center] [width] [height]
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with open(label_fpath, 'a') as f:
            f.write(label_str)
tid_curr = 0
tid_last = -1
for seq in seqs:
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

    for fid, tid, x, y, w, h, mark, label, _ in gt:
        if mark == 0 or not label == 1:
            continue
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))

        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with open(label_fpath, 'a') as f:
            f.write(label_str)