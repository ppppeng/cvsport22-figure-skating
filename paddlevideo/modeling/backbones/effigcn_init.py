import paddle
import math
#from paddlevideo.modeling.backbones.effigcn import EfficientGCN
# from paddlevideo.modeling.backbones.activations import *

import logging

from paddlevideo.loader.dataset.graphs import Graph
# from paddlevideo.loader.dataset.ntu_feeder import NTU_Feeder, NTU_Location_Feeder



class EffigcnInit():

    kwargs=None
    kwargs1=None
    def __init__(self,args):
        self.args=args
        self.init_dataset()
        self.init_model()

    def init_dataset(self):
        dataset_name = self.args.dataset.split('-')[0]
        dataset_args = self.args.dataset_args[dataset_name]
        dataset_args['debug'] = self.args.debug


        self.data_shape, self.num_class, self.A, self.parts = data_create(
            self.args.dataset, **dataset_args
        )

        kwargs = {
            'data_shape': self.data_shape,
            'num_class': 30,
            'A': paddle.Tensor(self.A),
            'parts': self.parts,
        }
        self.kwargs = kwargs
        EffigcnInit.kwargs=kwargs

    @staticmethod
    def getargs():
        return EffigcnInit.kwargs1


    def init_model(self):
        kwargs = {
            'data_shape': self.data_shape,
            'num_class':30,
            'A': paddle.Tensor(self.A),
            'parts': self.parts,
        }
        self.kwargs=kwargs

        self.create(self.args.model_type, **(self.args.model_args), **kwargs)


    def create(self,model_type, act_type, block_args, scale_args, **kwargs):
        __activations = {
            'relu': paddle.nn.ReLU(),
            'relu6': paddle.nn.ReLU6(),
            'hswish': paddle.nn.Hardswish(),
            'swish': paddle.nn.Swish(),
        }
        kwargs.update({
            'act': __activations[act_type],
            'block_args': self.rescale_block(block_args, scale_args, int(model_type[-1])),
        })

        EffigcnInit.kwargs=kwargs
        EffigcnInit.kwargs1=kwargs
        # return (EfficientGCN(**kwargs))

    def rescale_block(self,block_args, scale_args, scale_factor):
        channel_scaler = math.pow(scale_args[0], scale_factor)
        depth_scaler = math.pow(scale_args[1], scale_factor)
        new_block_args = []
        for [channel, stride, depth] in block_args:
            channel = max(int(round(channel * channel_scaler / 16)) * 16, 16)
            depth = int(round(depth * depth_scaler))
            new_block_args.append([channel, stride, depth])
        return new_block_args


__data_args = {
    'fsd': {'class': 30, 'shape': [3, 4, 500, 25, 1]},
}




def data_create(dataset, num_frame, inputs, **kwargs):
    graph = Graph(dataset)
    try:
        data_args = __data_args[dataset]
        data_args['shape'][0] = len(inputs)
        data_args['shape'][2] = num_frame
    except:
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.foramt(dataset))
        raise ValueError()

    kwargs.update({
        'inputs': inputs,
        'num_frame': num_frame,
        'connect_joint': graph.connect_joint,
    })


    return data_args['shape'], data_args['class'], graph.A, graph.parts


