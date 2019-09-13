from mxnet import autograd, gluon, gpu, init, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn, rnn


class KG(object):
    def __init__(self):
        pass
    
    def generate_dataset(self):
        self.entity_map = []
        self.relat_map = []

        # for i in range(3):
        #     self.entity_list[i] = nd.random_uniform(shape=(2, 2))
        
        # for i in range(3):
