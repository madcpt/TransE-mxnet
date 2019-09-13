from mxnet import autograd, gluon, gpu, init, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn, rnn


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


class Relation(object):
    def __init__(self, embedding, M1, M2):
        self.embedding = embedding
        self.M1 = M1
        self.M2 = M2
        

class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # self.dense = nn.Dense(2, activation='relu')
        self.M1 = self.params.get('M1', shape=(2, 2))
        self.M2 = self.params.get('M2', shape=(2, 2))
        
        self.A = self.params.get('A', shape=(2, 1))
        self.B = self.params.get('B', shape=(2, 1))
        self.C = self.params.get('C', shape=(2, 1))

        self.R1 = self.params.get('R1', shape=(2, 1))
        # self.R2 = self.params.get('R1', shape=(2, 1))

    def loss_function(self, h, r, t, M1, M2):
        ht = nd.dot(M1, h) + nd.dot(M2, t)
        L = nd.dot(r.T, nd.tanh(ht))
        return L

    def forward(self):
        logger.debug(self.A.data())
        logger.debug(self.B.data())
        logger.debug(self.C.data())
        logger.debug(self.R1.data())
        L = self.loss_function(self.A.data(), self.R1.data(), self.B.data(), self.M1.data(), self.M2.data())
        L = L + self.loss_function(self.C.data(), self.R1.data(), self.B.data(), self.M1.data(), self.M2.data())
        return L

if __name__ == '__main__':
    logger = Log(10, 'pg', False)
    
    # M1 = nd.random.uniform(shape=(2, 2))
    # M2 = nd.random.uniform(shape=(2, 2))

    # A = nd.random_uniform(shape=(2, 1))
    # B = nd.random_uniform(shape=(2, 1))
    # C = nd.random_uniform(shape=(2, 1))

    # R1 = nd.array([[1, 1]]).T

    Y = nd.zeros(shape=(1, 1))

    # A.attach_grad()
    # B.attach_grad()
    # C.attach_grad()
    # R1.attach_grad()

    net = FancyMLP()
    net.initialize()
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.3, 'momentum': 0, 'wd': 0})
    loss = gloss.L1Loss()

    for i in range(10):
        logger.info(i)
        # logger.debug(A)
        # logger.debug(B)
        # logger.debug(R1)
        # logger.debug(net.M1.data())
        # logger.debug(net.M2.data())
        with autograd.record():
            output = net()
            l = loss(output, Y).mean()
            print(l)
        l.backward()
        trainer.step(1)


    # logger.debug(net(X))
