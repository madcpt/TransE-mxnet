import math

import numpy as np
from mxnet import autograd, cpu, gluon, gpu, init, initializer, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
from mxnet.gluon.nn import Embedding

from DataLoader import DataLoader

loader = DataLoader('SMALL')
loader.load_all()
loader.preprocess(0, False)
# print(loader.train_triple)

class MLP(nn.Block):
    def __init__(self, *args):
        super(MLP, self).__init__(*args)
    
    def load(self, loader:DataLoader):
        self.train_triple = nd.array(loader.train_triple)
        self.test_triple = nd.array(loader.test_triple)
        self.train_triple_list = loader.train_triple
        self.test_triple_list = loader.test_triple
        self.dim = 2
        self.entity_size = loader.entity_size
        self.relation_size = loader.relation_size
        self.entity_embedding = Embedding(self.entity_size, self.dim, weight_initializer=init.Uniform(6.0/math.sqrt(self.dim)))
        self.relation_embedding = Embedding(self.relation_size, self.dim, weight_initializer=init.Uniform(6.0/math.sqrt(self.dim)))
        return
    
    def setup_corrupt_tail(self):
        self.corrupt_tail_list = []
        for i in range(self.entity_size):
            entity_good_list = []
            for j in range(self.relation_size):
                entity_good_list.append(list(range(self.entity_size)))
            self.corrupt_tail_list.append(entity_good_list)

        for train_triple in self.train_triple_list:
            self.corrupt_tail_list[train_triple[0]][train_triple[1]].remove(train_triple[2])
        for head in range(self.entity_size):
            for relation in range(self.relation_size):
                self.corrupt_tail_list[head][relation].remove(head)
    
    def setup_corrupt_head(self):
        self.corrupt_head_list = []
        for i in range(self.entity_size):
            entity_good_list = []
            for j in range(self.relation_size):
                entity_good_list.append(list(range(self.entity_size)))
            self.corrupt_head_list.append(entity_good_list)

        for train_triple in self.train_triple_list:
            self.corrupt_head_list[train_triple[2]][train_triple[1]].remove(train_triple[0])
        for tail in range(self.entity_size):
            for relation in range(self.relation_size):
                self.corrupt_head_list[tail][relation].remove(tail)

    def normalize_parameters(self, layer='entity'):
        if layer=='relation':
            for tag in list(self.relation_embedding.params):
                weight = self.relation_embedding.params[tag].data().detach()
                self.relation_embedding.params[tag].set_data(weight/weight.norm(axis=-1, keepdims=True))
                # print('Relation.grad()')
                # print(self.relation_embedding.params[tag].grad())
        if layer=='entity':
            for tag in list(self.entity_embedding.params):
                # print('*'*40)
                # print('Entity.grad()')
                # print(self.entity_embedding.params[tag].grad())
                weight = self.entity_embedding.params[tag].data().detach()
                self.entity_embedding.params[tag].set_data(weight/weight.norm(axis=-1, keepdims=True))
                # print(self.entity_embedding.params[tag].grad())
                # print('*'*40)
    
    def get_triple_embdeding(self, start, end, mode='train'):
        if mode == 'train':
            current_train_triple = self.train_triple[start: end]
            heads_idx = current_train_triple[:,0]
            heads_embedding = self.entity_embedding(heads_idx.reshape(len(heads_idx),1))
            relations_idx = current_train_triple[:,1]
            relations_embedding = self.relation_embedding(relations_idx.reshape(len(heads_idx),1))
            tails_idx = current_train_triple[:,2]
            tails_embedding = self.entity_embedding(tails_idx.reshape(len(heads_idx),1))
            # triple_embedding = nd.concat(heads_embedding, relations_embedding, tails_embedding)
            return (heads_embedding, relations_embedding, tails_embedding)

    def negative_sampling(self, start, end, max_round=20, sparse=False):
        '''Decrepted'''
        current_triple_set = self.train_triple_list[start:end]
        import random
        negative_sample = []
        if not sparse:
            for (head, relat, tail) in current_triple_set:
                choice = random.choice(self.corrupt_tail_list[head][relat])
                negative_sample.append((head, relat, tail, head, choice))
                choice = random.choice(self.corrupt_head_list[tail][relat])
                negative_sample.append((head, relat, tail, choice, tail))
        return negative_sample
    
    def get_corruput_triple_embdeding(self, start, end, mode='train'):
        if mode == 'train':
            current_train_currupt_triple = nd.array(self.negative_sampling(start, end))
            print('current_train_currupt_triple')
            print(current_train_currupt_triple)
            heads_idx = current_train_currupt_triple[:,0]
            heads_embedding = self.entity_embedding(heads_idx.reshape(len(heads_idx),1))
            relations_idx = current_train_currupt_triple[:,1]
            relations_embedding = self.relation_embedding(relations_idx.reshape(len(heads_idx),1))
            tails_idx = current_train_currupt_triple[:,2]
            tails_embedding = self.entity_embedding(tails_idx.reshape(len(heads_idx),1))
            neg_heads_idx = current_train_currupt_triple[:,3]
            neg_heads_embedding = self.entity_embedding(neg_heads_idx.reshape(len(heads_idx),1))
            neg_tails_idx = current_train_currupt_triple[:,4]
            neg_tails_embedding = self.entity_embedding(neg_tails_idx.reshape(len(heads_idx),1))
            # triple_embedding = nd.concat(heads_embedding, relations_embedding, tails_embedding)
            return (heads_embedding, relations_embedding, tails_embedding, neg_heads_embedding, neg_tails_embedding)
    
    def forward(self, start, end, margin=1, ord=ord):
        corrupt_triple = self.get_corruput_triple_embdeding(start, end)
        # print(nd.concat(*corrupt_triple))
        return f_loss(*corrupt_triple, margin=margin, ord=ord)
    
def distance(heads_embedding, relation_embedding, tails_embedding, ord=1):
    d = (heads_embedding+relation_embedding-tails_embedding).norm(ord=ord, axis=-1)
    return d

def f_loss(heads_embedding, relation_embedding, tails_embedding, 
                    neg_heads_embedding, neg_tails_embedding, margin=1, ord=1):
    d1 = distance(heads_embedding, relation_embedding, tails_embedding,  ord=ord)
    d2 = distance(neg_heads_embedding, relation_embedding, neg_tails_embedding, ord=ord)
    L = nd.maximum(margin + d1 - d2, 0)
    # print(d1)
    # print(d2)
    # print(L)
    return L


optimizer = 'sgd'
lr = {'learning_rate': 1e0}

loss = gloss.L1Loss()

net = MLP()
net.load(loader)
net.initialize()
print('setup_corrupt_tail')
net.setup_corrupt_tail()
print('setup_corrupt_head')
net.setup_corrupt_head()

trainer = gluon.Trainer(net.collect_params(), optimizer, lr)

net.normalize_parameters('entity')
net.normalize_parameters('relation')
triple_embedding = net.get_triple_embdeding(0, loader.train_triple_size, 'train')
print(nd.concat(*triple_embedding))
# d = distance(*triple_embedding)

print('Start iteration')
for i in range(1):
    print('*'*40)
    net.normalize_parameters('entity')
    with autograd.record():
        output = net(0, loader.train_triple_size, 0.2, 1)
    #     l = loss(output, nd.zeros(output.shape)).mean()
    # l.backward()
    output.backward()
    
    for tag in list(net.entity_embedding.params):
        print('Entity.data()')
        print(net.entity_embedding.params[tag].data())
        print('Entity.grad()')
        print(net.entity_embedding.params[tag].grad())

    # tag = list(net.relation_embedding.params)[0]
    # weight = net.relation_embedding.params[tag]
    # print(l)
    # print(weight.data())
    # print(weight.grad())
    trainer.step(len(output))
    # print(weight.data())
    # print(weight.grad())
    # print('*'*40)
    # for tag in list(net.entity_embedding.params):
    #     print('Entity.data()')
    #     print(net.entity_embedding.params[tag].data())
    # print('output')
    # print(output)

triple_embedding = net.get_triple_embdeding(0, loader.train_triple_size, 'train')
print(nd.concat(*triple_embedding))
l = distance(*triple_embedding)
print(l)
