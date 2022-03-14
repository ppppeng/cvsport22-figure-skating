# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from turtle import st
import paddle
import argparse
from paddlevideo.utils import get_config
from paddlevideo.tasks import train_model, train_model_multigrid, test_model, train_dali
from paddlevideo.utils import get_dist_info
import numpy as np
import random
from paddlevideo.modeling.backbones.effigcn_init import EffigcnInit


def predefined_model_list():
    agcn_model_list = [
        {'cfg_path': 'configs/recognition/agcn_fold/agcn_fsd0.yaml',
         'weight_path': './output/AGCN_fold0/AGCN_best.pdparams',
         'test_window_size': [350]},
        {'cfg_path': 'configs/recognition/agcn_fold/agcn_fsd1.yaml',
         'weight_path': './output/AGCN_fold1/AGCN_best.pdparams',
         'test_window_size': [350]},
        {'cfg_path': 'configs/recognition/agcn_fold/agcn_fsd2.yaml',
         'weight_path': './output/AGCN_fold2/AGCN_best.pdparams',
         'test_window_size': [350]},
        {'cfg_path': 'configs/recognition/agcn_fold/agcn_fsd3.yaml',
         'weight_path': './output/AGCN_fold3/AGCN_best.pdparams',
         'test_window_size': [350]},
        {'cfg_path': 'configs/recognition/agcn_fold/agcn_fsd4.yaml',
         'weight_path': './output/AGCN_fold4/AGCN_best.pdparams',
         'test_window_size': [350]}
    ]
    # ctrgcn_model_list = [
    #     {'cfg_path': 'configs/recognition/ctrgcn_fold/ctrgcn_fsd0.yaml',
    #      'weight_path': './output/CTRGCN_fold0/CTRGCN_best.pdparams',
    #      'test_window_size': [300, 350, 450]},
    #     {'cfg_path': 'configs/recognition/ctrgcn_fold/ctrgcn_fsd1.yaml',
    #      'weight_path': './output/CTRGCN_fold1/CTRGCN_best.pdparams',
    #      'test_window_size': [300, 350, 450]},
    #     {'cfg_path': 'configs/recognition/ctrgcn_fold/ctrgcn_fsd2.yaml',
    #      'weight_path': './output/CTRGCN_fold2/CTRGCN_best.pdparams',
    #      'test_window_size': [300, 350, 450]},
    #     {'cfg_path': 'configs/recognition/ctrgcn_fold/ctrgcn_fsd3.yaml',
    #      'weight_path': './output/CTRGCN_fold3/CTRGCN_best.pdparams',
    #      'test_window_size': [300, 350, 450]},
    #     {'cfg_path': 'configs/recognition/ctrgcn_fold/ctrgcn_fsd4.yaml',
    #      'weight_path': './output/CTRGCN_fold4/CTRGCN_best.pdparams',
    #      'test_window_size': [300, 350, 450]}
    # ]
    ctrgcn2_model_list = [
        {'cfg_path': 'configs/recognition/ctrgcn2_fold/ctrgcn_fsd0.yaml',
         'weight_path': './output/CTRGCN2_fold0/CTRGCN_best.pdparams',
         'test_window_size': [350]},
        {'cfg_path': 'configs/recognition/ctrgcn2_fold/ctrgcn_fsd1.yaml',
         'weight_path': './output/CTRGCN2_fold1/CTRGCN_best.pdparams',
         'test_window_size': [350]},
        {'cfg_path': 'configs/recognition/ctrgcn2_fold/ctrgcn_fsd2.yaml',
         'weight_path': './output/CTRGCN2_fold2/CTRGCN_best.pdparams',
         'test_window_size': [350]},
        {'cfg_path': 'configs/recognition/ctrgcn2_fold/ctrgcn_fsd3.yaml',
         'weight_path': './output/CTRGCN2_fold3/CTRGCN_best.pdparams',
         'test_window_size': [350]},
        {'cfg_path': 'configs/recognition/ctrgcn2_fold/ctrgcn_fsd4.yaml',
         'weight_path': './output/CTRGCN2_fold4/CTRGCN_best.pdparams',
         'test_window_size': [350]}
    ]
    stgcn_model_list = [
        {'cfg_path': 'configs/recognition/stgcn_fold/stgcn_fsd0.yaml',
         'weight_path': './output/STGCN_fold0/STGCN_best.pdparams',
         'test_window_size': [350]},
        {'cfg_path': 'configs/recognition/stgcn_fold/stgcn_fsd1.yaml',
         'weight_path': './output/STGCN_fold1/STGCN_best.pdparams',
         'test_window_size': [350]},
        {'cfg_path': 'configs/recognition/stgcn_fold/stgcn_fsd2.yaml',
         'weight_path': './output/STGCN_fold2/STGCN_best.pdparams',
         'test_window_size': [350]},
        {'cfg_path': 'configs/recognition/stgcn_fold/stgcn_fsd3.yaml',
         'weight_path': './output/STGCN_fold3/STGCN_best.pdparams',
         'test_window_size': [350]},
        {'cfg_path': 'configs/recognition/stgcn_fold/stgcn_fsd4.yaml',
         'weight_path': './output/STGCN_fold4/STGCN_best.pdparams',
         'test_window_size': [350]},
    ]
    effigcn_model_list = [
        {'cfg_path': 'configs/recognition/effigcn_fold/effigcn_fsd0.yaml',
         'weight_path': './output/EFFGCN_fold0/EFFGCN_best.pdparams',
         'test_window_size': [500]},
        {'cfg_path': 'configs/recognition/effigcn_fold/effigcn_fsd1.yaml',
         'weight_path': './output/EFFGCN_fold1/EFFGCN_best.pdparams',
         'test_window_size': [500]},
        {'cfg_path': 'configs/recognition/effigcn_fold/effigcn_fsd2.yaml',
         'weight_path': './output/EFFGCN_fold2/EFFGCN_best.pdparams',
         'test_window_size': [500]},
        {'cfg_path': 'configs/recognition/effigcn_fold/effigcn_fsd3.yaml',
         'weight_path': './output/EFFGCN_fold3/EFFGCN_best.pdparams',
         'test_window_size': [500]},
        {'cfg_path': 'configs/recognition/effigcn_fold/effigcn_fsd4.yaml',
         'weight_path': './output/EFFGCN_fold4/EFFGCN_best.pdparams',
         'test_window_size': [500]},
    ]
    # models = agcn_model_list + ctrgcn_model_list + ctrgcn2_model_list + stgcn_model_list + effigcn_model_list
    models = agcn_model_list +  ctrgcn2_model_list + stgcn_model_list 
    # models=effigcn_model_list
    return models


def parse_args():
    parser = argparse.ArgumentParser("PaddleVideo train script")
    # parser.add_argument('-c',
    #                     '--config',
    #                     type=str,
    #                     default='configs/example.yaml',
    #                     help='config file path')
    parser.add_argument('-o',
                        '--override',
                        action='append',
                        default=[],
                        help='config options to be overridden')
    parser.add_argument('--test',
                        action='store_true',
                        help='whether to test a model')
    parser.add_argument('--train_dali',
                        action='store_true',
                        help='whether to use dali to speed up training')
    parser.add_argument('--multigrid',
                        action='store_true',
                        help='whether to use multigrid training')
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        help='weights for finetuning or testing')
    parser.add_argument('--fleet',
                        action='store_true',
                        help='whether to use fleet run distributed training')
    parser.add_argument('--amp',
                        action='store_true',
                        help='whether to open amp training.')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument('-g', '--gpu', default="3", type=str)

    args = parser.parse_args()
    return args


def main():
    dataset_dir='/home/data2/lzp_dataset/skating'
    args = parse_args()
    # cfg = get_config(args.config, overrides=args.override)

    _, world_size = get_dist_info()
    parallel = world_size != 1
    if parallel:
        paddle.distributed.init_parallel_env()

    model_list = predefined_model_list()
    print("************************************************************")
    for model in model_list:
        print(model['cfg_path'])
    print("************************************************************")

    for model in model_list:
        np.random.seed(42)
        random.seed(2021)
        paddle.seed(102)
        cfg = get_config(model['cfg_path'], overrides=args.override)
        cfg['DATASET']['train']['file_path']=cfg['DATASET']['train']['file_path'].replace('DATASET_DIR',dataset_dir)
        cfg['DATASET']['train']['label_path']=cfg['DATASET']['train']['label_path'].replace('DATASET_DIR',dataset_dir)
        cfg['DATASET']['valid']['file_path']=cfg['DATASET']['valid']['file_path'].replace('DATASET_DIR',dataset_dir)
        cfg['DATASET']['valid']['label_path']=cfg['DATASET']['valid']['label_path'].replace('DATASET_DIR',dataset_dir)
        cfg['DATASET']['test']['file_path']=cfg['DATASET']['test']['file_path'].replace('DATASET_DIR',dataset_dir)
        
        cfg['gpu'] = args.gpu
        print("*************" + model['cfg_path'] + "*******************")

        if (cfg.MODEL.backbone.name == 'EfficientGCN'):
            EffigcnInit(cfg)  # 初始化EfficientGCN参数

        if args.test:
            test_model(cfg, weights=model['weight_path'], parallel=parallel, test_window_size=model['test_window_size'])
        else:
            train_model(cfg, weights=args.weights, parallel=parallel, validate=args.validate, use_fleet=args.fleet,
                        amp=args.amp)

    # if args.test:
    #     test_model(cfg, weights=args.weights, parallel=parallel)
    # elif args.train_dali:
    #
    #     train_dali(cfg, weights=args.weights, parallel=parallel)
    # elif args.multigrid:
    #     train_model_multigrid(cfg, world_size, validate=args.validate)
    # else:
    #     train_model(cfg,
    #                 weights=args.weights,
    #                 parallel=parallel,
    #                 validate=args.validate,
    #                 use_fleet=args.fleet,
    #                 amp=args.amp)


if __name__ == '__main__':
    main()
