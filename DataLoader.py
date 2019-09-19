import collections


class DataLoader(object):
    def __init__(self, dataset='WN18', *args):
        self.train_path = './data/{}/train.txt'.format(dataset)
        self.valid_path = './data/{}/valid.txt'.format(dataset)
        self.test_path = './data/{}/test.txt'.format(dataset)
        self.entity_map_path = './data/{}/entity_map.db'.format(dataset)
        self.relation_map_path = './data/{}/relation_map.db'.format(dataset)
        self.train_list = []
        self.valid_list = []
        self.test_list = []
        self.entity_map = {}
        self.relation_map = {}
        self.entity_size = 0
        self.relation_size = 0
        self.train_triple = []
        self.valid_triple = []
        self.test_triple = []
        self.train_triple_size = 0
        self.valid_triple_size = 0
        self.test_triple_size = 0
        

    def load_all(self):
        with open(self.train_path, 'r') as f:
            lines = f.readlines()
        self.train_list = [line.split() for line in lines]
        print('Trainset size: {}'.format(len(self.train_list)))

        with open(self.valid_path, 'r') as f:
            lines = f.readlines()
        self.valid_list = [line.split() for line in lines]
        print('Validset size: {}'.format(len(self.valid_list)))
        
        with open(self.test_path, 'r') as f:
            lines = f.readlines()
        self.test_list = [line.split() for line in lines]
        print('Testset size: {}'.format(len(self.test_list)))
    
    def counter_filter(self, raw_dataset, count=1):
        counter = collections.Counter([tk for tk in raw_dataset])
        counter = dict(filter(lambda x: x[1] >= count, counter.items()))
        return counter

    def preprocess(self, filter_occurance=5, init=False):
        '''Preprocess the dataset.

        Parameters
        ----------
        filter_occurance : int
            Only entities that occur no fewer than 'filter_occurance' will be included 
            (occurring in either head or tail is qualified).
        init : bool, default False
            Whether to recreate entity2idx map and relation2idx map.
           '''
        all_list = [*self.train_list,*self.valid_list,*self.test_list]
        entity_list = []
        relation_list = []
        for triple in all_list:
            entity_list.append(triple[0])
            relation_list.append(triple[1])
            entity_list.append(triple[2])
        entity_counter = self.counter_filter(entity_list, filter_occurance)
        relation_counter = self.counter_filter(relation_list, 1)
        if init:
            for (i, entity) in enumerate(entity_counter.keys()):
                self.entity_map[entity] = i
            for (i, relation) in enumerate(relation_counter.keys()):
                self.relation_map[relation] = i
            with open(self.entity_map_path, 'w') as f:
                f.write(str(self.entity_map))
            with open(self.relation_map_path, 'w') as f:
                f.write(str(self.relation_map))
        else:
            print('Reading {}'.format(self.entity_map_path))
            with open(self.entity_map_path, 'r') as f:
                self.entity_map = eval(f.read())
            print('Reading {}'.format(self.relation_map_path))
            with open(self.relation_map_path, 'r') as f:
                self.relation_map = eval(f.read())
        self.entity_size = len(self.entity_map.keys())
        self.relation_size = len(self.relation_map.keys())
        
        print('Entity_size: {}'.format(self.entity_size))
        print('Relation_size: {}'.format(self.relation_size))

        self.train_triple = [(self.entity_map[i[0]], self.relation_map[i[1]], self.entity_map[i[2]]) 
                                for i in self.train_list
                                if (i[0] in self.entity_map.keys() and i[1] in self.relation_map.keys() and
                                    i[2] in self.entity_map.keys())]
        self.valid_triple = [(self.entity_map[i[0]], self.relation_map[i[1]], self.entity_map[i[2]]) 
                                for i in self.valid_list
                                if (i[0] in self.entity_map.keys() and i[1] in self.relation_map.keys() and
                                    i[2] in self.entity_map.keys())]
        self.test_triple = [(self.entity_map[i[0]], self.relation_map[i[1]], self.entity_map[i[2]]) 
                                for i in self.test_list
                                if (i[0] in self.entity_map.keys() and i[1] in self.relation_map.keys() and
                                    i[2] in self.entity_map.keys())]
        self.train_triple_size = len(self.train_triple)
        self.valid_triple_size = len(self.valid_triple)
        self.test_triple_size = len(self.test_triple)
        
        
        

if __name__ == "__main__":
    loader = DataLoader(dataset='WN18')
    loader.load_all()
    loader.preprocess(1, True)
