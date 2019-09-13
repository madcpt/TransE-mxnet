import matplotlib.pyplot as plt
import numpy as np
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn, rnn
from mxnet.gluon.nn import LayerNorm


def draw(p1): 
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
    def __init__(self, entity_size=0, relation_size=0, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.entity_list = list(range(entity_size))
        self.relation_list = list(range(relation_size))
        self.triple_set = []
        self.norm_layer = LayerNorm(scale = True)

        # self.dense = nn.Dense(2, activation='relu')
        # self.M1 = self.params.get_constant('M1', [[1,1],[1,1]])
        # self.M2 = self.params.get_constant('M2', [[1,1],[1,1]])
        
        # self.A = self.params.get('A', shape=(2, 1))
        # self.B = self.params.get('B', shape=(2, 1))
        
        # self.entity_list = []
        # self.entity_list.append(self.params.get('A', shape=(2, 1)))
        # self.entity_list.append(self.params.get('B', shape=(2, 1)))

        # self.C = self.params.get('C', shape=(2, 1))

        # self.R1 = self.params.get('R1', shape=(2, 1))
        # self.R2 = self.params.get('R1', shape=(2, 1))

    def add_relation_list(self, new_relation_list:[Relation]):
        for relat in new_relation_list:
            if type(self.relation_list[relat.idx]) is int:
                # self.relation_list[relat.idx] = relat.embedding
                self.relation_list[relat.idx] = self.params.get(relat.tag, shape=(1,3), init=init.Normal(1))
            if type(self.entity_list[relat.head.idx]) is int:
                # self.entity_list[relat.head.idx] = relat.head.embedding
                self.entity_list[relat.head.idx] = self.params.get(relat.head.tag, shape=(1,3), init=init.Normal(1))
            if type(self.entity_list[relat.tail.idx]) is int:
                # self.entity_list[relat.tail.idx] = relat.tail.embedding
                self.entity_list[relat.tail.idx] = self.params.get(relat.tail.tag, shape=(1,3), init=init.Normal(1))
            triple = (relat.head.idx, relat.idx, relat.tail.idx)
            self.triple_set.append(triple)
            
    # def normalize(self):
    #     for (i, e) in enumerate(self.entity_list):
    #         if type(e) is not int:
    #             self.entity_list[i] = e.data() / e.data().norm()
    #             # print(e.data())

    def take_log(self, h, r, t, M1 = None, M2 = None):
        logger.debug('h: ' + str(h))
        logger.debug('t: ' + str(r))
        logger.debug('r' + str(t))
        if M1 is not None:
            logger.debug('M1' + str(M1))
        if M2 is not None:
            logger.debug('M2' + str(M2))

    # def loss_function(self, h, r, t, M1, M2):
    #     self.take_log(h, r, t, M1, M2)
    #     ht = nd.dot(M1, h) + nd.dot(M2, t)
    #     L = nd.dot(r.T, nd.tanh(ht))
    #     return L

    def loss_function(self, h, r, t):
        # self.take_log(h, r, t)
        L = h + r - t
        return L

    # def norm_layer(self, x):
    #     return x/x.norm()
        

    def forward(self, start=0, end=0):
        (h_i, r_i, t_i) = self.triple_set[start]
        h = self.entity_list[h_i].data()
        r = self.relation_list[r_i].data()
        t = self.entity_list[t_i].data()
        h = self.norm_layer(h)
        r = self.norm_layer(r)
        t = self.norm_layer(t)
        # h = self.norm_layer(h).reshape((1, *h.shape))
        # r = self.norm_layer(r).reshape((1, *r.shape))
        # t = self.norm_layer(t).reshape((1, *t.shape))
        L = self.loss_function(h, r, t)
        return L

    def backward(self):
        print("bingo")
        return 
    
    def dump(self, path, loss):
        with open(path, 'w') as f:
            for (h_i, r_i, t_i) in self.triple_set:
                h = self.entity_list[h_i].data()
                r = self.relation_list[r_i].data()
                t = self.entity_list[t_i].data()
                h = self.norm_layer(h)
                r = self.norm_layer(r)
                t = self.norm_layer(t)
                f.write('{}-{}-{} \n{} - {} - {} - {}\n\n'.format(str(h_i), str(r_i), str(t_i), str(h), str(r), str(t), str(loss(self.loss_function(h, r, t), nd.zeros(shape=(1,3))))))


def build_dataset():
    relation_list = []
    A = Entity(0, 'A', nd.ones(shape=(1,3)))
    B = Entity(1, 'B', nd.ones(shape=(1,3)))
    C = Entity(2, 'C', nd.ones(shape=(1,3)))
    relation_list.append(Relation(0, 'r1', A, B, nd.ones(shape=(1,3))))
    relation_list.append(Relation(1, 'r2', B, C, nd.ones(shape=(1,3))))
    relation_list.append(Relation(1, 'r2', C, A, nd.ones(shape=(1,3))))
    return relation_list

if __name__ == '__main__':
    logger = Log(10, 'pg', False)
    
    # M1 = nd.random_normal(loc=1, shape=(2, 2))
    # M2 = nd.random_normal(loc=1, shape=(2, 2))

    # A = nd.random_normal(loc=1, shape=(2, 1))
    # B = nd.random_normal(loc=1, shape=(2, 1))
    # C = nd.random_uniform(shape=(2, 1))

    # R1 = nd.array([[1, 1]]).T

    Y = nd.zeros(shape=(1,3))

    # A.attach_grad()
    # B.attach_grad()
    # C.attach_grad()
    # R1.attach_grad()

    # M1.attach_grad()
    # M2.attach_grad()

    data = build_dataset()

    net = FancyMLP(3, 3)
    net.add_relation_list(data)
    net.initialize()

    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': 0.01})
    loss = gloss.L2Loss()

    p = []
    for i in range(100):
        # logger.info('*'*40)
        # logger.debug(A)
        # logger.debug(B)
        # logger.debug(R1)
        # logger.debug(net.M1.data())
        # logger.debug(net.M2.data())
        # net.normalize()
        l = 0
        with autograd.record():
            # for j in range(3):
            output = net(0, 2)
            l = l + loss(output, Y)
            output = net(1, 2)
            l = l + loss(output, Y)
            output = net(2, 2)
            l = l + loss(output, Y)
            logger.info('epoch {}: {}'.format(str(i), str(l.asscalar())))
            p.append(l.asscalar())
        l.backward()
        trainer.step(3)
    draw(p)
    net.dump('relations.txt',loss)

    # logger.debug(net(X))
