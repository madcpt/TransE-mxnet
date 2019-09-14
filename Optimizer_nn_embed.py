import numpy as np
from mxnet import autograd, context, cpu, gluon, gpu, init, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn, rnn
from mxnet.gluon.nn import Embedding, LayerNorm
from mxnet.initializer import Initializer


def draw(p1): 
    import matplotlib.pyplot as plt
    plt.figure('Draw')
    plt.plot(p1)  
    plt.draw()  
    plt.pause(3)  
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
    def __init__(self, entity_size=0, relation_size=0, entity_dim=2, relation_dim=2, sampling_rate=0.2, ctx=None, logger=None, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.entity_list = list(range(entity_size))
        self.relation_list = list(range(relation_size))
        self.triple_set = []
        # self.norm_layer = LayerNorm(scale = True)
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.entity_embedding = Embedding(entity_size, entity_dim)
        self.relation_embedding = Embedding(relation_size, relation_dim)
        self.sampling_rate=sampling_rate
        self.ctx = ctx
        self.logger = logger

    def add_relation_list(self, new_relation_list:[Relation]):
        for relat in new_relation_list:
            if type(self.entity_list[relat.head.idx]) is int:
                self.entity_list[relat.head.idx] = self.params.get(relat.head.tag, shape=(1,self.entity_dim), init=init.Uniform(1))
            if type(self.relation_list[relat.idx]) is int:
                self.relation_list[relat.idx] = self.params.get(relat.tag, shape=(1,self.relation_dim), init=init.One())
            if type(self.entity_list[relat.tail.idx]) is int:
                self.entity_list[relat.tail.idx] = self.params.get(relat.tail.tag, shape=(1,self.entity_dim), init=init.Uniform(1))
            triple = (relat.head.idx, relat.idx, relat.tail.idx)
            self.triple_set.append(triple)
            
    # def normalize(self):
    #     for (i, e) in enumerate(self.entity_list):
    #         if type(e) is not int:
    #             self.entity_list[i] = e.data() / e.data().norm()
    #             # print(e.data())

    # def take_log(self, h, r, t, M1 = None, M2 = None):
    #     logger.debug('h: ' + str(h))
    #     logger.debug('t: ' + str(r))
    #     logger.debug('r' + str(t))
    #     if M1 is not None:
    #         logger.debug('M1' + str(M1))
    #     if M2 is not None:
    #         logger.debug('M2' + str(M2))

    # def loss_function(self, h, r, t, M1, M2):
    #     self.take_log(h, r, t, M1, M2)
    #     ht = nd.dot(M1, h) + nd.dot(M2, t)
    #     L = nd.dot(r.T, nd.tanh(ht))
    #     return L

    def distance(self, h, r, t):
        # self.take_log(h, r, t)
        D = h + r - t
        return D

    def loss_function(self, h, r, t, h_hat, t_hat, gamma=0.5):
        # print(self.distance(h,r,t) - self.distance(h_hat,r,t_hat))
        # print(nd.array(gamma + self.distance(h,r,t) - self.distance(h_hat,r,t_hat), self.ctx))
        # print(nd.maximum(nd.array(gamma + self.distance(h,r,t) - self.distance(h_hat,r,t_hat), self.ctx), 0))
        L = nd.maximum(nd.array(gamma + self.distance(h,r,t) - self.distance(h_hat,r,t_hat), self.ctx), 0)
        return L

    def negative_sampling(self, triple_set:[int], sampling_rate=0.1):
        import random
        negative_sample = []
        for head_idx in list(range(self.entity_size)):
            for tail_idx in list(range(self.entity_size)):
                if head_idx == tail_idx:
                    continue
                exist_relat = []
                for (head, relat, tail) in triple_set:
                    if relat not in exist_relat and (head_idx, relat, tail_idx) not in triple_set:
                        exist_relat.append(relat)
                        if random.random()<sampling_rate:
                            negative_sample.append((head, relat, tail, head_idx, tail_idx))
        return negative_sample

    def norm_layer(self, x, dim):
        return (x/x.norm(axis=-1, keepdims=True))

    def forward(self, start=0, end=0):
        # (h_i, r_i, t_i) = self.triple_set[start]
        new_triple_set = self.negative_sampling(self.triple_set, self.sampling_rate)
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

        h = self.norm_layer(h, self.entity_dim)
        t = self.norm_layer(t, self.entity_dim)
        # r = self.norm_layer(r, self.relation_dim)
        h_hat = self.norm_layer(h_hat, self.entity_dim)
        t_hat = self.norm_layer(t_hat, self.entity_dim)
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
    
    def dump(self, path, loss, ctx):
        with open(path, 'w') as f:
            h_i = nd.array([triple[0] for triple in self.triple_set], ctx=ctx)
            r_i = nd.array([triple[1] for triple in self.triple_set], ctx=ctx)
            t_i = nd.array([triple[2] for triple in self.triple_set], ctx=ctx)
            # self.logger.debug(h_i)     
            h = self.entity_embedding(h_i)
            r = self.relation_embedding(r_i)
            t = self.entity_embedding(t_i)
            # self.logger.debug(h)
            h = self.norm_layer(h, self.entity_dim)
            # r = self.norm_layer(r, self.relation_dim)
            t = self.norm_layer(t, self.entity_dim)
            f.write('{}-{}-{} \n{} - {} - {} - {} - {}\n\n'.format(
                    str(h_i), str(r_i), str(t_i), str(h), str(r), str(t), 
                    '',''))
            # f.write('{}-{}-{} \n{} - {} - {} - {} - {}\n\n'.format(
            #         str(h_i), str(r_i), str(t_i), str(h), str(r), str(t), 
            #         str(self.distance(h, r, t)), str(self.distance(h, r, t).abs().sum())))


def build_dataset(entity_dim=2, relation_dim=2, ctx=None):
    relation_list = []
    A = Entity(0, 'A', nd.ones(shape=(1,entity_dim), ctx=ctx))
    B = Entity(1, 'B', nd.ones(shape=(1,entity_dim), ctx=ctx))
    C = Entity(2, 'C', nd.ones(shape=(1,entity_dim), ctx=ctx))
    relation_list.append(Relation(0, 'r1', A, B, nd.ones(shape=(1,relation_dim), ctx=ctx)))
    relation_list.append(Relation(0, 'r1', B, C, nd.ones(shape=(1,relation_dim), ctx=ctx)))
    relation_list.append(Relation(1, 'r2', C, A, nd.ones(shape=(1,relation_dim), ctx=ctx)))
    return relation_list

if __name__ == '__main__':
    logger = Log(10, 'pg', False)
    
    # M1 = nd.random_normal(loc=1, shape=(2, 2))
    # M2 = nd.random_normal(loc=1, shape=(2, 2))

    # A = nd.random_normal(loc=1, shape=(2, 1))
    # B = nd.random_normal(loc=1, shape=(2, 1))
    # C = nd.random_uniform(shape=(2, 1))

    # R1 = nd.array([[1, 1]]).T

    entity_size = 3
    relation_size = 2
    entity_dim = 2
    relation_dim = 2

    ctx = gpu(0)

    Y = nd.zeros(shape=(entity_size,entity_dim), ctx=ctx)

    # A.attach_grad()
    # B.attach_grad()
    # C.attach_grad()
    # R1.attach_grad()

    # M1.attach_grad()
    # M2.attach_grad()

    data = build_dataset(entity_dim=entity_dim, relation_dim=relation_dim, ctx=ctx)

    net = FancyMLP(entity_size, relation_size,
                        entity_dim, relation_dim,
                        # sampling_rate=0.2, 
                        ctx=ctx, logger=logger)
    net.add_relation_list(data)
    net.initialize(force_reinit=True, ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.01})
    loss = gloss.L1Loss()

    p = []

    for i in range(100):
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
            l = l.sum()
            logger.info('epoch {}: {}'.format(str(i), str(l.asscalar())))
            p.append(l.asscalar())
            # print(l)
        l.backward()
        trainer.step(relation_size, True)
    draw(p)
    net.dump('relations.txt', loss, ctx)

    # logger.debug(net(X))
