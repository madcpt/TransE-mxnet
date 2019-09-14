import math

import numpy as np
from mxnet import autograd, context, cpu, gluon, gpu, init, initializer, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn, rnn
from mxnet.gluon.nn import Embedding, LayerNorm


def draw(p1): 
    import matplotlib.pyplot as plt
    plt.figure('Draw')
    plt.plot(p1)  
    plt.draw()  
    plt.pause(1)  
    plt.savefig("easyplot01.jpg")
    plt.close()

def Log(default_level, log_path, live_stream):
    import logging
    
    logger = logging.getLogger(__name__)
    logger.setLevel(default_level)
    formatter = logging.Formatter("%(levelname)s:%(module)s:%(lineno)d:%(message)s")

    if log_path is not None:
        import time
        time_tag = time.strftime("%Y-%a-%b-%d-%H-%M-%S", time.localtime())
        file_path = 'log/{}-{}.log'.format(log_path, time_tag)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if live_stream is not None and live_stream == True:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)
    return logger

class Entity(object):
    def __init__(self, idx, tag, embedding):
        self.idx = idx
        self.tag = tag
        self.embedding = embedding

class Relation(object):
    def __init__(self, idx, tag, head, tail, embedding):
        self.idx = idx
        self.tag = tag
        self.head = head
        self.tail = tail
        self.embedding = embedding

        

class FancyMLP(nn.Block):
    def __init__(self, entity_size=0, relation_size=0, entity_dim=2, relation_dim=2, negative_sampling_rate=0.5, margin=0.1, ctx=None, logger=None, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.entity_list = list(range(entity_size))
        self.relation_list = list(range(relation_size))
        self.triple_set = []
        # self.norm_layer = LayerNorm(scale = True)
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.entity_embedding = Embedding(entity_size, entity_dim, weight_initializer=initializer.Uniform(6.0/math.sqrt(entity_dim)))
        self.relation_embedding = Embedding(relation_size, relation_dim, weight_initializer=initializer.Uniform(6.0/math.sqrt(entity_dim)))
        self.negative_sampling_rate=negative_sampling_rate
        self.margin = margin
        self.ctx = ctx
        self.logger = logger

    def add_relation_list(self, new_relation_list:[Relation]):
        for relat in new_relation_list:
            # if type(self.entity_list[relat.head.idx]) is int:
            #     self.entity_list[relat.head.idx] = self.params.get(relat.head.tag, shape=(1,self.entity_dim), init=init.Uniform(1))
            # if type(self.relation_list[relat.idx]) is int:
            #     self.relation_list[relat.idx] = self.params.get(relat.tag, shape=(1,self.relation_dim), init=init.Uniform(1))
            # if type(self.entity_list[relat.tail.idx]) is int:
            #     self.entity_list[relat.tail.idx] = self.params.get(relat.tail.tag, shape=(1,self.entity_dim), init=init.Uniform(1))
            triple = (relat.head.idx, relat.idx, relat.tail.idx)
            self.triple_set.append(triple)
            

    def distance(self, h, r, t, ord=1):
        # self.take_log(h, r, t)
        D = (h + r - t).norm(ord=ord, axis=-1)
        return D

    def loss_function(self, h, r, t, h_hat, t_hat):
        # print(self.distance(h,r,t) - self.distance(h_hat,r,t_hat))
        # print(nd.array(margin + self.distance(h,r,t) - self.distance(h_hat,r,t_hat), self.ctx))
        # print(nd.maximum(nd.array(margin + self.distance(h,r,t) - self.distance(h_hat,r,t_hat), self.ctx), 0))
        L = nd.maximum(nd.array(self.margin + self.distance(h,r,t) - self.distance(h_hat,r,t_hat), self.ctx), 0)
        # print(self.distance(h,r,t) - self.distance(h_hat,r,t_hat))
        return L

    # def negative_sampling(self, triple_set:[int], negative_sampling_rate=0.5):
    #     import random
    #     negative_sample = []
    #     for head_idx in list(range(self.entity_size)):
    #         for tail_idx in list(range(self.entity_size)):
    #             if head_idx == tail_idx:
    #                 continue
    #             exist_relat = []
    #             for (head, relat, tail) in triple_set:
    #                 if (head_idx, relat, tail_idx) not in triple_set:
    #                     if random.random()<negative_sampling_rate:
    #                         negative_sample.append((head, relat, tail, head_idx, tail_idx))
    #                         exist_relat.append(relat)
    #     self.logger.debug(negative_sample)
    #     return negative_sample
    def negative_sampling(self, triple_set:[int], negative_sampling_rate=0.5):
        import random
        negative_sample = []
        for (head, relat, tail) in triple_set:
            sample_source = []
            for head_idx in list(range(self.entity_size)):
                for tail_idx in list(range(self.entity_size)):
                    if head_idx == tail_idx:
                        continue
                    if (head_idx, relat, tail_idx) not in triple_set:
                        sample_source.append((head_idx, tail_idx))
            choice = random.choice(sample_source)
            negative_sample.append((head, relat, tail, *choice))
            # TODO
        self.logger.debug(negative_sample)
        return negative_sample

    def norm_layer(self, x, dim):
        # return (x/x.norm(axis=-1, keepdims=True))
        return x

    def normalize_relation_parameters(self):
        for tag in list(self.relation_embedding.params):
            # logger.debug(self.relation_embedding.params[tag].data())
            weight = self.relation_embedding.params[tag]
            self.relation_embedding.params[tag].set_data(weight.data()/weight.data().norm(axis=-1, keepdims=True))
    
    def normalize_entity_parameters(self):
        for tag in list(self.entity_embedding.params):
            # logger.debug(self.entity_embedding.params[tag].data())
            weight = self.entity_embedding.params[tag]
            self.entity_embedding.params[tag].set_data(weight.data()/weight.data().norm(axis=-1, keepdims=True))
            # logger.debug(self.entity_embedding.params[tag].data())
        # for (i,entity) in enumerate(self.entity_embedding):
        #     if type(entity) is not int:
        #         self.entity_list[i] = (entity.data()/entity.data().norm(axis=-1, keepdims=True))

    def forward(self, start=0, end=0):
        # (h_i, r_i, t_i) = self.triple_set[start]
        new_triple_set = []
        while new_triple_set == []:
            new_triple_set = self.negative_sampling(self.triple_set, self.negative_sampling_rate)
        h_i = nd.array([triple[0] for triple in new_triple_set], ctx=self.ctx)
        r_i = nd.array([triple[1] for triple in new_triple_set], ctx=self.ctx)
        t_i = nd.array([triple[2] for triple in new_triple_set], ctx=self.ctx)
        h_hat_i = nd.array([triple[3] for triple in new_triple_set], ctx=self.ctx)
        t_hat_i = nd.array([triple[4] for triple in new_triple_set], ctx=self.ctx)
        # logger.debug(h_i)
        # logger.debug(t_i)
        # logger.debug(h_hat_i)
        # logger.debug(t_hat_i)

        h = self.entity_embedding(h_i)
        t = self.entity_embedding(t_i)
        r = self.relation_embedding(r_i)
        h_hat = self.entity_embedding(h_hat_i)
        t_hat = self.entity_embedding(t_hat_i)
        # logger.debug(h)
        # logger.debug(t)
        # logger.debug(h_hat)
        # logger.debug(t_hat)

        # h = self.norm_layer(h, self.entity_dim)
        # t = self.norm_layer(t, self.entity_dim)
        # # r = self.norm_layer(r, self.relation_dim)
        # h_hat = self.norm_layer(h_hat, self.entity_dim)
        # t_hat = self.norm_layer(t_hat, self.entity_dim)
        # logger.debug(h)
        # logger.debug(t)
        # logger.debug(h_hat)
        # logger.debug(t_hat)
        
        L = self.loss_function(h, r, t, h_hat, t_hat)
        self.logger.debug(L)
        return L

    def backward(self):
        print("bingo")
        return 
    
    def dump(self, path, loss):
        with open(path, 'w') as f:
            h_i = nd.array([triple[0] for triple in self.triple_set], ctx=self.ctx)
            r_i = nd.array([triple[1] for triple in self.triple_set], ctx=self.ctx)
            t_i = nd.array([triple[2] for triple in self.triple_set], ctx=self.ctx)

            h = self.entity_embedding(h_i)
            r = self.relation_embedding(r_i)
            t = self.entity_embedding(t_i)

            h = self.norm_layer(h, self.entity_dim)
            t = self.norm_layer(t, self.entity_dim)

            f.write('{}-{}-{} \n{} - {} - {} - {} - {}\n\n'.format(
                    str(h_i), str(r_i), str(t_i), str(h), str(r), str(t), 
                    str(self.distance(h, r, t)), str(self.distance(h, r, t).abs().sum())))


def build_dataset(entity_dim=2, relation_dim=2, ctx=None):
    relation_list = []
    A = Entity(0, 'A', nd.ones(shape=(1,entity_dim), ctx=ctx))
    B = Entity(1, 'B', nd.ones(shape=(1,entity_dim), ctx=ctx))
    C = Entity(2, 'C', nd.ones(shape=(1,entity_dim), ctx=ctx))
    D = Entity(3, 'D', nd.ones(shape=(1,entity_dim), ctx=ctx))
    E = Entity(4, 'E', nd.ones(shape=(1,entity_dim), ctx=ctx))
    relation_list.append(Relation(0, 'r1', A, B, nd.ones(shape=(1,relation_dim), ctx=ctx)))
    relation_list.append(Relation(0, 'r1', B, C, nd.ones(shape=(1,relation_dim), ctx=ctx)))
    relation_list.append(Relation(0, 'r1', C, D, nd.ones(shape=(1,relation_dim), ctx=ctx)))
    relation_list.append(Relation(0, 'r1', D, E, nd.ones(shape=(1,relation_dim), ctx=ctx)))
    relation_list.append(Relation(1, 'r2', C, B, nd.ones(shape=(1,relation_dim), ctx=ctx)))
    # relation_list.append(Relation(1, 'r2', B, A, nd.ones(shape=(1,relation_dim), ctx=ctx)))
    return relation_list

if __name__ == '__main__':
    logger = Log(10, 'pg', False)

    entity_size = 5
    relation_size = 2
    entity_dim = 10
    relation_dim = 10

    ctx = gpu(0)

    Y = nd.zeros(shape=(entity_size,entity_dim), ctx=ctx)

    data = build_dataset(entity_dim=entity_dim, relation_dim=relation_dim, ctx=ctx)

    net = FancyMLP(entity_size, relation_size,
                        entity_dim, relation_dim,
                        # negative_sampling_rate=0.3, 
                        margin=1,
                        ctx=ctx, logger=logger)
    net.add_relation_list(data)
    net.initialize(force_reinit=True, ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.01})

    net.normalize_relation_parameters()

    loss = gloss.L2Loss()

    p = []

    for i in range(300):
        net.normalize_entity_parameters()
        # logger.info('*'*40)
        # logger.debug(A)
        # logger.debug(B)
        # logger.debug(R1)
        # logger.debug(net.M1.data())
        # logger.debug(net.M2.data())
        # net.normalize()
        with autograd.record():
            # for j in range(3):
            output = net(0, 2)
            logger.debug(output)
            l = loss(output, nd.zeros(output.shape, ctx=ctx))
            logger.debug(l)
            logger.info('epoch {}: {}'.format(str(i), str(l.sum().asscalar())))
            l = l.mean()
            p.append(l.asscalar())
            # print(l)
        l.backward()
        trainer.step(1)
    net.dump('relations.txt', loss)
    draw(p)

    # logger.debug(net(X))
