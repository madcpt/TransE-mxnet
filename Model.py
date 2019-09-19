import math
import time

import numpy as np
from mxnet import autograd, cpu, gluon, gpu, init, initializer, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
from mxnet.gluon.nn import Embedding

from DataLoader import DataLoader
from Entity import Entity
from Relation import Relation
from utils.draw import draw
from utils.Log import Log


class TransE(nn.Block):
    def __init__(self, entity_size=0, relation_size=0, entity_dim=2, relation_dim=2, 
                negative_sampling_rate=0.5, sample_raw_negative=True, margin=0.1, 
                ctx=None, logger=None, param_path='./param/', sparse=True, **kwargs):
        super(TransE, self).__init__(**kwargs)
        # self.entity_list = list(range(entity_size))
        # self.relation_list = list(range(relation_size))
        self.train_triple_set = []
        self.valid_triple_set = []
        self.test_triple_set = []
        self.train_triple_set_nd = np.array([])
        self.train_triple_head_set_nd = np.array([])
        # self.norm_layer = LayerNorm(scale = True)
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.entity_embedding = Embedding(entity_size, entity_dim, 
                    weight_initializer=initializer.Uniform(6.0/math.sqrt(entity_dim)))
        self.relation_embedding = Embedding(relation_size, relation_dim, 
                    weight_initializer=initializer.Uniform(6.0/math.sqrt(relation_dim)))
        self.negative_sampling_rate=negative_sampling_rate
        self.sample_raw_negative=sample_raw_negative
        self.margin = margin
        self.ctx = ctx
        self.logger = logger
        self.param_path = param_path
        self.sparse = sparse
        self.training_log = []

    def load_relation_data(self, relation_data:list, mode='complex', type='train'):
        if mode == 'simple':
            for relat in relation_data:
                triple = (relat.head.idx, relat.idx, relat.tail.idx)
                if type=='train':
                    self.train_triple_set.append(triple)
                    self.train_triple_set_nd = nd.array(self.train_triple_set, ctx=self.ctx, dtype='int32')
                    self.train_triple_head_set_nd = self.train_triple_set_nd[:,0].reshape(len(self.train_triple_set_nd), 1)
                if type=='valid':
                    self.valid_triple_set.append(triple)
                if type=='test':
                    self.test_triple_set.append(triple)
        if mode == 'complex':
            if type=='train':
                self.train_triple_set.extend(relation_data)
                self.train_triple_set_nd = nd.array(self.train_triple_set, ctx=self.ctx, dtype='int32')
                self.train_triple_head_set_nd = self.train_triple_set_nd[:,0].reshape(len(self.train_triple_set_nd), 1)
            if type=='valid':
                self.valid_triple_set.extend(relation_data)
            if type=='test':
                self.test_triple_set.extend(relation_data)

    def distance(self, h, r, t, ord=1):
        # self.take_log(h, r, t)
        D = (h + r - t).norm(ord=ord, axis=-1)
        return D

    def loss_function(self, h, r, t, h_hat, t_hat):
        d1 = self.distance(h,r,t)
        d2 = self.distance(h_hat,r,t_hat)
        L = nd.maximum(nd.array(self.margin + d1 - d2, self.ctx), 0)
        # L = self.distance(h,r,t)
        # print(self.distance(h,r,t) - self.distance(h_hat,r,t_hat))
        return L

    def negative_sampling(self, start, end, negative_sampling_rate=0.5, max_round=20, sparse=True):
        '''Decrepted'''
        triple_set = self.train_triple_set[start:end]
        start = time.time()
        import random
        negative_sample = []
        if sparse:
            fetch_time = 0.0
            valid_time = 0.0
            for (head, relat, tail) in triple_set:
                # for i in range(max_round):
                #     head_idx = random.randint(0, self.entity_size-1)
                #     if head_idx == tail:
                #         continue
                #     if (head_idx, relat, tail) not in triple_set:
                #         negative_sample.append((head, relat, tail, head_idx, tail))
                #         break

                for i in range(max_round):
                    t1 = time.time()
                    tail_idx = random.randint(0, self.entity_size-1)
                    t2 = time.time()
                    if head == tail_idx:
                        continue
                    if (self.sample_raw_negative) or (head, relat, tail_idx) not in triple_set:
                        negative_sample.append((head, relat, tail, head, tail_idx))
                        break
                    t3 = time.time()
                    fetch_time += t2 - t1
                    valid_time += t3 - t2
            self.logger.error("Negative sampling time: {} {} {}".format(str(fetch_time),
                        str(valid_time),str(time.time()-start)))
        else:
            #TODO raw
            for (head, relat, tail) in triple_set:
                sample_source = []
                for head_idx in list(range(self.entity_size)):
                    if head_idx == tail:
                        continue
                    if (head_idx, relat, tail) not in triple_set:
                        sample_source.append((head_idx, tail))
                choice = random.choice(sample_source)
                negative_sample.append((head, relat, tail, *choice))

                sample_source = []
                for tail_idx in list(range(self.entity_size)):
                    if head == tail_idx:
                        continue
                    if (head, relat, tail_idx) not in triple_set:
                        sample_source.append((head, tail_idx))
                choice = random.choice(sample_source)
                negative_sample.append((head, relat, tail, *choice))            # TODO
        self.logger.info('Negative sampling inside time: {}'.format(str(time.time()-start)))
        return negative_sample
    
    def negative_sampling_new(self, start, end, negative_sampling_rate=0.5, max_round=20, sparse=True):
        # TODO raw, sparse
        t1 = time.time()
        negative_sample = []
        clipped_triple_set_nd = self.train_triple_set_nd[start:end]
        clipped_triple_head_set_idx = self.train_triple_head_set_nd[start:end]
        t2 = time.time()
        if sparse:
            all_tails_idx = nd.random.randint(0, self.entity_size,
                                        shape=(len(clipped_triple_set_nd),1), ctx=self.ctx, dtype='int32')
            t3 = time.time()
            negative_sample = nd.concat(clipped_triple_set_nd, 
                                        clipped_triple_head_set_idx, 
                                        clipped_triple_head_set_idx, dim=-1)
            negative_sample = list(negative_sample.asnumpy())
            # print(t2-t1)
            # print(t3-t2)
            # print(time.time() - t3)
            # print(negative_sample)
        return negative_sample

    def norm_layer(self, x, dim):
        # return (x/x.norm(axis=-1, keepdims=True))
        return x

    def normalize_relation_parameters(self, ord=2):
        for tag in list(self.relation_embedding.params):
            # logger.debug(self.relation_embedding.params[tag].data())
            weight = self.relation_embedding.params[tag].data().detach()
            self.relation_embedding.params[tag].set_data(weight/weight.norm(ord=ord, axis=-1, keepdims=True))
            self.relation_embedding.params[tag].zero_grad()
            # print(self.relation_embedding.params[tag].grad())
    
    def normalize_entity_parameters(self, ord=2):
        for tag in list(self.entity_embedding.params):
            # logger.debug(self.entity_embedding.params[tag].data())
            weight = self.entity_embedding.params[tag].data().detach()
            self.entity_embedding.params[tag].set_data(weight/weight.norm(ord=ord, axis=-1, keepdims=True))
            self.entity_embedding.params[tag].zero_grad()
            # print(self.entity_embedding.params[tag].grad())

    def forward(self, start=0, end=10):
        # (h_i, r_i, t_i) = self.train_triple_set[start]
        t1 = time.time()
        new_train_triple_set = self.negative_sampling(
                                    start, min(end, len(self.train_triple_set)), 
                                    self.negative_sampling_rate, 
                                    sparse=self.sparse)
        while len(new_train_triple_set) == 0:
            # TODO: possible bug
            new_train_triple_set = self.negative_sampling(
                                        start, min(end, len(self.train_triple_set)), 
                                        self.negative_sampling_rate, 
                                        sparse=self.sparse)
            self.logger.warning('Negative sampling failed. repeat the process.')
        if len(self.train_triple_set[start:end]) != len(new_train_triple_set):
            self.logger.warning('samping flaw: {} -> {}'.format(str(len(self.train_triple_set[start:end])), str(len(new_train_triple_set))))
        # logger.debug(new_train_triple_set)
        t2 = time.time()
        h_i = nd.array([triple[0] for triple in new_train_triple_set], ctx=self.ctx)
        r_i = nd.array([triple[1] for triple in new_train_triple_set], ctx=self.ctx)
        t_i = nd.array([triple[2] for triple in new_train_triple_set], ctx=self.ctx)
        h_hat_i = nd.array([triple[3] for triple in new_train_triple_set], ctx=self.ctx)
        t_hat_i = nd.array([triple[4] for triple in new_train_triple_set], ctx=self.ctx)
        t3 = time.time()
        # logger.debug(h_i)
        # logger.debug(t_i)
        # logger.debug(h_hat_i)
        # logger.debug(t_hat_i)

        h = self.entity_embedding(h_i)
        t = self.entity_embedding(t_i)
        r = self.relation_embedding(r_i)
        h_hat = self.entity_embedding(h_hat_i)
        t_hat = self.entity_embedding(t_hat_i)
        t4 = time.time()
        # logger.debug(h)
        # logger.debug(t)
        # logger.debug(h_hat)
        # logger.debug(t_hat)
        
        L = self.loss_function(h, r, t, h_hat, t_hat)
        t5 = time.time()
        # self.logger.warning('forward time details: {} {} {} {}'.format(
        #     str(t2 - t1),
        #     str(t3 - t2),
        #     str(t4 - t3),
        #     str(t5 - t4)
        # ))
        # self.logger.debug(L)
        return L

    def backward(self):
        print("bingo")
        return 
    
    def save_embeddings(self, model_name):
        self.entity_embedding.save_parameters('{}{}_entity.params'.format(
                        self.param_path, model_name))
        self.relation_embedding.save_parameters('{}{}_relation.params'.format(
                        self.param_path, model_name))
        with open('{}{}_log.stat'.format(self.param_path, model_name), 'w') as f:
            f.write(str(self.training_log))
                
    def load_embeddings(self, model_name):
        self.entity_embedding.load_parameters('{}{}_entity.params'.format(
                        self.param_path, model_name), ctx=self.ctx)
        self.relation_embedding.load_parameters('{}{}_relation.params'.format(
                        self.param_path, model_name), ctx=self.ctx)
        with open('{}{}_log.stat'.format(self.param_path, model_name), 'r') as f:
            self.training_log = eval(f.read())    
    
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
    
    def get_triple_embdeding(self, start, end, mode='train'):
        if mode == 'train':
            current_train_triple = self.train_triple_set_nd[start: end]
            heads_idx = current_train_triple[:,0]
            heads_embedding = self.entity_embedding(heads_idx.reshape(len(heads_idx),1))
            relations_idx = current_train_triple[:,1]
            relations_embedding = self.relation_embedding(relations_idx.reshape(len(heads_idx),1))
            tails_idx = current_train_triple[:,2]
            tails_embedding = self.entity_embedding(tails_idx.reshape(len(heads_idx),1))
            # triple_embedding = nd.concat(heads_embedding, relations_embedding, tails_embedding)
            return (heads_embedding, relations_embedding, tails_embedding)
    
    def predict_with_h_r_old(self, head_list, relation_list, k=3, ord=1):
        '''Deprecated'''
        heads = self.entity_embedding(head_list)
        relations = self.relation_embedding(relation_list)
        tails = self.entity_embedding(nd.array(list(range(self.entity_size)), ctx=self.ctx))
        # print(heads)
        # print(relations)
        # print(tails)
        prediction = []
        prediction_d = []
        # print('step1')
        for i in range(len(head_list)):
            # print('Start {}'.format(str(i)))
            prediction_i = []
            prediction_d_i = []
            candidates = self.distance(heads[i], relations[i], tails)
            # print('step2')
            candidates = candidates.asnumpy()
            # print(candidates)
            ceil = candidates.max()+1
            # print(ceil)
            # print('step3')
            for j in range(len(k)):
                min_idx = candidates.argmin()
                # print(candidates[min_idx])
                if candidates[min_idx] != ceil:
                    prediction_i.append(min_idx)
                    prediction_d_i.append(candidates[min_idx])
                    candidates[min_idx] = ceil
            prediction.append(prediction_i)
            prediction_d.append(prediction_d_i)
            # print(prediction_i)
            # print('step-end')
        return prediction, prediction_d
        
    def predict_with_h_r(self, head_list, relation_list, tail_list, k=[3], ord=1):
        heads = self.entity_embedding(head_list)
        relations = self.relation_embedding(relation_list)
        targets = self.entity_embedding(tail_list)
        tails = self.entity_embedding(nd.array(list(range(self.entity_size)), ctx=self.ctx))

        distance = self.distance(heads, relations, targets)
        
        # print(heads)
        # print(relations)
        # print(tails)
        # print(targets)
        # print(distance)
        # prediction = []
        # prediction_d = []
        # print('step1')
        all_hit = [0]*len(k)
        mean_rank = 0
        for i in range(len(head_list)):
            # print('Start {}'.format(str(i)))
            # prediction_i = []
            # prediction_d_i = []
            candidates = self.distance(heads[i], relations[i], tails)
            flag = candidates <= distance[i]
            count = flag.sum().asscalar()
            mean_rank += count
            for j in range(len(k)):
                if count<=k[j]:
                    all_hit[j] += 1
            # print('step2')
            # candidates = candidates.asnumpy()
            # # print(candidates)
            # ceil = candidates.max()+1
            # # print(ceil)
            # # print('step3')
            # for j in range(k):
            #     min_idx = candidates.argmin()
            #     # print(candidates[min_idx])
            #     if candidates[min_idx] != ceil:
            #         prediction_i.append(min_idx)
            #         prediction_d_i.append(candidates[min_idx])
            #         candidates[min_idx] = ceil
            # prediction.append(prediction_i)
            # prediction_d.append(prediction_d_i)
            # print(prediction_i)
            # print('step-end')
        mean_rank /= len(head_list)
        return (all_hit, mean_rank)
    
    def get_rank(self, target_value, prediction_d):
        for i in range(len(prediction_d)):
            if target_value <= prediction_d[i]:
                return i
        return -1

    def get_old_log(self):
        return self.training_log
    
    def get_loss_trend(self):
        return [i[1] for i in self.training_log]
    
    def add_training_log(self, epoch_num:int, total_loss:float, epoch_time:float):
        if len(self.training_log) == epoch_num:
            self.training_log.append((epoch_num, total_loss, epoch_time))

    def evaluate(self, mode='test', k=[3], ord=1):
        if mode == 'test':
            head_list = nd.array([i[0] for i in self.test_triple_set], ctx=self.ctx)
            relation_list = nd.array([i[1] for i in self.test_triple_set], ctx=self.ctx)
            tail_list = nd.array([i[2] for i in self.test_triple_set], ctx=self.ctx)
            print("Test set loaded.")
        if mode == 'valid':
            head_list = nd.array([i[0] for i in self.valid_triple_set], ctx=self.ctx)
            relation_list = nd.array([i[1] for i in self.valid_triple_set], ctx=self.ctx)
            tail_list = nd.array([i[2] for i in self.valid_triple_set], ctx=self.ctx)
        total = len(tail_list)
        # hit = 0
        t1 = time.time()
        # prediction, prediction_d = self.predict_with_h_r(head_list, relation_list, tail_list, k, ord)
<<<<<<< HEAD
        (all_hit, mean_rank) = self.predict_with_h_r(head_list, relation_list, tail_list, k, ord)
=======
        all_hit = self.predict_with_h_r(head_list, relation_list, tail_list, k, ord)
>>>>>>> e636dff8755eedb714f6e440ee8c41b42c97462c
        t2 = time.time()
        print('Prediction completed, time used: {}'.format(str(t2-t1)))
        # print(head_list)
        # print(relation_list)
        # print(tail_list)
        # print(prediction)
        # print(prediction_d)
        # for i in range(len(tail_list)):
        #     total += 1
        #     is_hit = tail_list[i] in prediction[i]
        #     # self.logger.debug('predict: {} - {} - {}'.format(tail_list[i],  prediction[i], prediction_d[i]))
        #     if is_hit:
        #         hit += 1
        #     else:
        #         h = self.entity_embedding(head_list[i])
        #         r = self.relation_embedding(relation_list[i])
        #         t = self.entity_embedding(tail_list[i])
        #         target_value = self.distance(h, r, t).asscalar()
        #         # print('target: {} rank: {} prediction: {}'.format(
        #         #         target_value,
        #         #         self.get_rank(target_value, prediction_d[i]),
        #         #         str(prediction_d[i])))
        #     if i%100 == 0:
        #         print('Evaluation: {}/{}: {}'.format(str(i), str(len(tail_list)), str(is_hit)))
        t3 = time.time()
        print('Evaluation time used: {}'.format(str(t3-t2)))
<<<<<<< HEAD
        print('Mean_rank: {}'.format(mean_rank))
=======
>>>>>>> e636dff8755eedb714f6e440ee8c41b42c97462c
        # for (head, relation, tail) in self.test_triple_set:
        #     total += 1
        #     prediction = self.predict_with_h_r(head, relation, k)
        #     self.logger.debug('{} - {}'.format(tail, prediction))
        #     if tail in [i[0] for i in prediction]:
        #         hit += 1
        return (total, all_hit)


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
    relation_list.append(Relation(1, 'r2', B, A, nd.ones(shape=(1,relation_dim), ctx=ctx)))
    relation_list.append(Relation(1, 'r2', C, B, nd.ones(shape=(1,relation_dim), ctx=ctx)))
    relation_list.append(Relation(1, 'r2', D, C, nd.ones(shape=(1,relation_dim), ctx=ctx)))
    relation_list.append(Relation(1, 'r2', E, D, nd.ones(shape=(1,relation_dim), ctx=ctx)))
    return relation_list


# if __name__ == '__main__':
#     logger = Log(10, 'pg', False)

#     local = False

#     if local:
#         ctx = gpu(0)
#         param_path = './param/'
#         model_name = 'simple'
#         mode = 'simple'
#         isTrain = True
#         isContinueTrain = False
#         sparse = False
#         margin = 0.3
#         epoch_num = 3000
#         k = 1
#     else:
#         ctx = gpu(1)
#         param_path = './param/'
#         model_name = 'WN'
#         mode = 'complex'
#         isTrain = True
#         isContinueTrain = False
#         sparse = True
#         margin=10
#         epoch_num = 30000
#         k = 20

#     if mode ==  'simple':
#         entity_size = 5
#         relation_size = 2
#         entity_dim = 2
#         relation_dim = 2
#         batch_size = 3
#         train_data = build_simple_dataset(entity_dim=entity_dim, 
#                                 relation_dim=relation_dim, ctx=ctx)
#         train_triple_size = len(train_data)
#         valid_data = train_data
#         valid_triple_size = train_triple_size
#         test_data = build_simple_dataset(entity_dim=entity_dim, 
#                                 relation_dim=relation_dim, ctx=ctx)
#         test_triple_size = len(test_data)
#         total_batch_num = math.ceil(train_triple_size/batch_size)
#     elif mode == 'complex':
#         loader = DataLoader()
#         print('Start loading data from {}'.format(loader.train_path))
#         print('Start loading data from {}'.format(loader.valid_path))
#         print('Start loading data from {}'.format(loader.test_path))
#         loader.load_all()
#         print('Start preprocessing...')
#         loader.preprocess(filter_occurance=1)
#         entity_size = loader.entity_size
#         relation_size = loader.relation_size
#         train_triple_size = loader.train_triple_size
#         entity_dim = 50
#         relation_dim = 50
#         batch_size = 500
#         total_batch_num = math.ceil(train_triple_size/batch_size)
#         train_data = loader.train_triple
#         valid_data = loader.valid_triple
#         test_data = loader.test_triple
#         print('Loading completed')
    
#     net = TransE(entity_size, relation_size,
#                         entity_dim, relation_dim,
#                         # negative_sampling_rate=0.3, 
#                         margin=margin,
#                         ctx=ctx, logger=logger, sparse=sparse, param_path=param_path)
#     loss = gloss.L1Loss()

#     if isTrain:
#         net.load_relation_data(train_data, mode=mode, type='train')
#         net.load_relation_data(valid_data, mode=mode, type='valid')
#         if isContinueTrain:
#             print('Loading embeddings')
#             net.load_embeddings(model_name=model_name)
#         else:
#             print('Initializing embeddings...')
#             net.initialize(force_reinit=True, ctx=ctx)
#             net.normalize_relation_parameters()
#         print('Setting up trainer...')
#         trainer = gluon.Trainer(net.collect_params(), 'sgd',
#                                 {'learning_rate': 0.1})


#         p = []
#         all_start = time.time()
#         print('Start iteration:')

#         for epoch in range(epoch_num):
#             epoch_loss = 0
#             epoch_start = time.time()
#             net.normalize_entity_parameters()
#             checkpoint = time.time()
#             # print('normalization completed, time used: {}'.format(str(checkpoint-epoch_start)))
#             # logger.info('*'*40)
#             for current_batch_num in range(total_batch_num):
#                 # print('current batch: {}'.format(str(current_batch_num)))
#                 with autograd.record():
#                     t1 = time.time()
#                     output = net(current_batch_num*batch_size, 
#                                 current_batch_num*batch_size+batch_size)
#                     # print(output)
#                     t2 = time.time()
#                     l = loss(output, nd.zeros(output.shape, ctx=ctx))
#                     t3 = time.time()
#                 batch_total_loss = l.sum().asscalar()
#                     # logger.debug(l)
#                     # l = l.mean()
#                 # print(t2-t1)
#                 # print(t3-t2)

#                 # print('epoch {} batch {}/{} completed, time used: {}, epoch time remaining: {}'.format(
#                 #         str(epoch),
#                 #         str(current_batch_num),
#                 #         str(total_batch_num),
#                 #         str(time.time()-epoch_start),
#                 #         str((time.time()-epoch_start)/(current_batch_num+1)*(total_batch_num-current_batch_num-1))))
                
#                 checkpoint = time.time()
#                 l.backward()
#                 # print('backward time: {}'.format(str(time.time()-checkpoint)))
#                 checkpoint = time.time()
#                 trainer.step(batch_size)
#                 # print('step time: {}'.format(str(time.time()-checkpoint)))
#                 checkpoint = time.time()
#                 batch_info = 'epoch {} batch {} time: {} \n\t total loss: {},\n\t avg loss: {}'.format(
#                         str(epoch),
#                         str(current_batch_num), 
#                         str(time.time() - t1),
#                         str(batch_total_loss),
#                         str(batch_total_loss/batch_size))
#                 # logger.info(batch_info)
#                 # print(batch_info)
#                 epoch_loss += batch_total_loss
#             p.append(epoch_loss/len(train_data)) 
#             #TODO
#             epoch_info = 'epoch {} time: {} \n\t total loss: {},\n\t avg loss: {}'.format(
#                     str(epoch), 
#                     str(time.time() - epoch_start),
#                     str(epoch_loss),
#                     str(epoch_loss / train_triple_size))
#             logger.info(epoch_info)
#             if (epoch+1) % 100 == 0:
#                 print('Auto-save parameters.')
#                 print(epoch_info)
#                 net.save_embeddings(model_name=model_name)
#                 checkpoint = time.time()
#                 (total, hit) = net.evaluate(mode='valid', k=k)
#                 net.logger.debug('Evaluation time: {} accuracy: {}'.format(str(time.time()-checkpoint), str(hit*1.0/total)))
#                 print('Evaluation time: {} accuracy: {}'.format(str(time.time()-checkpoint), str(hit*1.0/total)))
#                 if hit*1.0/total > 0.5:
#                     break


#             # net.save_parameters('{}model.params'.format(param_path))
#         net.save_embeddings(model_name=model_name)
#         # net.dump('relations.txt', loss)
#         draw(p)
#     else:
#         net.load_embeddings(model_name=model_name)
#         net.load_relation_data(test_data, mode=mode, type='test')
#         print('Initializing model...')
#         net.initialize(force_reinit=True, ctx=ctx)
#         (total, hit) = net.evaluate(k=100)
#         print(total)
#         print(hit)
#         print(hit/total)
#         # TODO

#     # for i in range(5):
#     #     print(net.predict_with_h_r(i,0))
#     # for i in range(5):
#     #     print(net.predict_with_h_r(i,1))
