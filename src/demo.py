from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trackeval import eval_once
from trackeval.utils import print_hota
from trackeval.utils import extract_dict

from AFLink.AppFreeLink import *

logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size) #视频加载相较于图片加载会降低HOTA值
    # data_root = os.path.join(opt.data_dir, 'sportsmot/images/val')
    # dataloader = datasets.LoadImages(osp.join(data_root, 'v_0kUtTtmLaJA_c005', 'img1'), opt.img_size)
    filename,ext=os.path.splitext(os.path.split(opt.input_video)[1])
    result_filename = os.path.join(result_root, filename+'.txt')
    frame_rate = 25

    if opt.use_AFLink:
        AFLink_model = PostLinker()
        AFLink_model.load_state_dict(torch.load(opt.path_AFLink))
        AFLink_dataset = LinkData('', '')

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, filename+'_frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])
    if opt.use_AFLink:
        linker = AFLink(
            path_in=result_filename,
            path_out=result_filename,
            model=AFLink_model,
            dataset=AFLink_dataset,
            thrT=(-10, 30),  # (-10, 30) for CenterTrack, FairMOT, TransTrack.
            thrS=75,
            thrP=0.10  # 0.10 for CenterTrack, FairMOT, TransTrack.
        )
        linker.link()
    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, filename+'_results.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    # demo(opt)

    #指标计算
    trackeval_dict = eval_once(
        "MOT_CHALLENGE_2D",
        ["HOTA","CLEAR","IDENTITY"],
        [
            ["../seq_results/hrnet18_number-v_0kUtTtmLaJA_c005/gt.txt", "../seq_results/hrnet18_number-v_0kUtTtmLaJA_c005/results.txt"],
        ])
    hota_dict = extract_dict(trackeval_dict)
    CLEAR_dict = (trackeval_dict['MotChallenge2DBox']['dataset_train']['seq_1']['pedestrian']['CLEAR'])
    Identity_dict = (trackeval_dict['MotChallenge2DBox']['dataset_train']['seq_1']['pedestrian']['Identity'])
    result_dict = {}
    result_dict['HOTA'] = hota_dict['HOTA']
    result_dict['DetA'] = hota_dict['DetA']
    result_dict['AssA'] = hota_dict['AssA']
    result_dict['MOTA'] = CLEAR_dict['MOTA']
    result_dict['IDF1'] = Identity_dict['IDF1']
    print(result_dict)
