import math

import numpy as np
from mxnet import autograd, context, cpu, gluon, gpu, init, initializer, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn, rnn
from mxnet.gluon.nn import Embedding, LayerNorm

from DataLoader import DataLoader
from Entity import Entity
from Relation import Relation
from utils.draw import draw
from utils.Log import Log


class FancyMLP(nn.Block):
    def __init__(self, entity_size=0, relation_size=0, entity_dim=2, relation_dim=2, negative_sampling_rate=0.5, margin=0.1, ctx=None, logger=None, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # self.entity_list = list(range(entity_size))
        # self.relation_list = list(range(relation_size))
        self.train_triple_set = []
        self.valid_triple_set = []
        self.test_triple_set = []
        # self.norm_layer = LayerNorm(scale = True)
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.entity_embedding = Embedding(entity_size, entity_dim, 
                    weight_initializer=initializer.Uniform(6.0/math.sqrt(entity_dim)))
        self.relation_embedding = Embedding(relation_size, relation_dim, 
                    weight_initializer=initializer.Uniform(6.0/math.sqrt(entity_dim)))
        self.negative_sampling_rate=negative_sampling_rate
        self.margin = margin
        self.ctx = ctx
        self.logger = logger

    def load_relation_data(self, relation_data:list, mode='complex', type='train'):
        if mode == 'complex':
            for relat in relation_data:
                triple = (relat.head.idx, relat.idx, relat.tail.idx)
                if type=='train':
                    self.train_triple_set.append(triple)
                if type=='valid':
                    self.valid_triple_set.append(triple)
                if type=='test':
                    self.test_triple_set.append(triple)
        if mode == 'simple':
            if type=='train':
                self.train_triple_set.extend(relation_data)
            if type=='valid':
                self.valid_triple_set.extend(relation_data)
            if type=='test':
                self.test_triple_set.extend(relation_data)
            

    def distance(self, h, r, t, ord=1):
        # self.take_log(h, r, t)
        D = (h + r - t).norm(ord=ord, axis=-1)
        return D

    def loss_function(self, h, r, t, h_hat, t_hat):
        L = nd.maximum(nd.array(self.margin + self.distance(h,r,t) -
                         self.distance(h_hat,r,t_hat), self.ctx), 0)
        # print(self.distance(h,r,t) - self.distance(h_hat,r,t_hat))
        return L

    def negative_sampling(self, train_triple_set:[int], negative_sampling_rate=0.5):
        import random
        negative_sample = []
        for (head, relat, tail) in train_triple_set:
            sample_source = []
            for head_idx in list(range(self.entity_size)):
                if head_idx == tail:
                    continue
                if (head_idx, relat, tail) not in train_triple_set:
                    sample_source.append((head_idx, tail))
            choice = random.choice(sample_source)
            negative_sample.append((head, relat, tail, *choice))

            sample_source = []
            for tail_idx in list(range(self.entity_size)):
                if head == tail_idx:
                    continue
                if (head, relat, tail_idx) not in train_triple_set:
                    sample_source.append((head, tail_idx))
            choice = random.choice(sample_source)
            negative_sample.append((head, relat, tail, *choice))
            # TODO
        # self.logger.debug(negative_sample)
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

    def forward(self, start=0, end=0):
        # (h_i, r_i, t_i) = self.train_triple_set[start]
        new_train_triple_set = []
        while new_train_triple_set == []:
            new_train_triple_set = self.negative_sampling(self.train_triple_set, self.negative_sampling_rate)
        h_i = nd.array([triple[0] for triple in new_train_triple_set], ctx=self.ctx)
        r_i = nd.array([triple[1] for triple in new_train_triple_set], ctx=self.ctx)
        t_i = nd.array([triple[2] for triple in new_train_triple_set], ctx=self.ctx)
        h_hat_i = nd.array([triple[3] for triple in new_train_triple_set], ctx=self.ctx)
        t_hat_i = nd.array([triple[4] for triple in new_train_triple_set], ctx=self.ctx)
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
        
        L = self.loss_function(h, r, t, h_hat, t_hat)
        # self.logger.debug(L)
        return L

    def backward(self):
        print("bingo")
        return 
    
    def dump(self, path, loss):
        with open(path, 'w') as f:
            h_i = nd.array([triple[0] for triple in self.train_triple_set], ctx=self.ctx)
            r_i = nd.array([triple[1] for triple in self.train_triple_set], ctx=self.ctx)
            t_i = nd.array([triple[2] for triple in self.train_triple_set], ctx=self.ctx)

            h = self.entity_embedding(h_i)
            r = self.relation_embedding(r_i)
            t = self.entity_embedding(t_i)

            h = self.norm_layer(h, self.entity_dim)
            t = self.norm_layer(t, self.entity_dim)

            f.write('{}-{}-{} \n{} - {} - {} - {} - {}\n\n'.format(
                    str(h_i), str(r_i), str(t_i), str(h), str(r), str(t), 
                    str(self.distance(h, r, t)), str(self.distance(h, r, t).abs().sum())))
    
    def predict_with_h_r(self, head_idx, relation_idx, k=3, ord=2):
        head = net.entity_embedding(nd.array([head_idx], ctx=net.ctx))
        relation = net.relation_embedding(nd.array([relation_idx], ctx=net.ctx))
        tails = net.entity_embedding(nd.array(list(range(net.entity_size)), ctx=self.ctx))
        candidates = []
        for tail in tails:
            candidates.append(net.distance(head, relation, tail, ord=ord).asscalar())
        candidates = np.array(candidates)
        # net.logger.info(candidates)
        prediction = []
        max = candidates.max()
        for i in range(k):
            min_idx = candidates.argmin()
            if candidates[min_idx] != max:
                prediction.append((min_idx, candidates[min_idx]))
                candidates[min_idx] = max
        return prediction


def build_simple_dataset(entity_dim=2, relation_dim=2, ctx=None):
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
    return relation_list


if __name__ == '__main__':
    logger = Log(10, 'pg', False)

    # entity_size = 5
    # relation_size = 2
    # entity_dim = 10
    # relation_dim = 10

    ctx = gpu(0)

    # Y = nd.zeros(shape=(entity_size,entity_dim), ctx=ctx)

    loader = DataLoader()
    print('Start loading data from {}'.format(loader.train_path))
    print('Start loading data from {}'.format(loader.valid_path))
    print('Start loading data from {}'.format(loader.test_path))
    loader.load_all()
    print('Start preprocessing...')
    loader.preprocess()

    entity_size = loader.entity_size
    relation_size = loader.relation_size
    entity_dim = 50
    relation_dim = 50

    train_data = loader.train_triple
    print('Loading completed')

    # data = build_simple_dataset(entity_dim=entity_dim, relation_dim=relation_dim, ctx=ctx)

    net = FancyMLP(entity_size, relation_size,
                        entity_dim, relation_dim,
                        # negative_sampling_rate=0.3, 
                        margin=2,
                        ctx=ctx, logger=logger)
    net.load_relation_data(train_data, mode='simple', type='train')
    print('Initializing model...')
    net.initialize(force_reinit=True, ctx=ctx)

    print('Setting up trainer...')
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.01})

    net.normalize_relation_parameters()

    loss = gloss.L2Loss()

    p = []

    print('Start iteration:')
    
    for i in range(1):
        net.normalize_entity_parameters()
        # logger.info('*'*40)
        with autograd.record():
            output = net(0, 2)
            # logger.debug(output)
            l = loss(output, nd.zeros(output.shape, ctx=ctx))
            # logger.debug(l)
            logger.info('epoch {}: {}'.format(str(i), str(l.sum().asscalar())))
            print('epoch {}: \n\t total loss: {},\n\t avg loss: {}'.format(str(i), 
                    str(l.sum().asscalar()),
                    str(l.mean().asscalar())))
            l = l.mean()
            p.append(l.asscalar())
        l.backward()
        trainer.step(1)
    net.dump('relations.txt', loss)
    draw(p)

    for i in range(5):
        print(net.predict_with_h_r(i,0))
    for i in range(5):
        print(net.predict_with_h_r(i,1))
