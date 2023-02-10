cd src
python train.py mot --exp_id sportsmot_data_dla34 --arch 'dla_34' --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/sportsmot_data.json'
cd .