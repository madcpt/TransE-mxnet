import collections


class DataLoader(object):
    def __init__(self, *args):
        self.train_path = './data/WN18/train.txt'
        self.valid_path = './data/WN18/valid.txt'
        self.test_path = './data/WN18/test.txt'
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
        self.train_triple_size = []
        self.valid_triple_size = []
        self.test_triple_size = []
        

    def load_all(self):
        with open(self.train_path, 'r') as f:
            lines = f.readlines()
        self.train_list = [line.split() for line in lines]
        print(len(self.train_list))

        with open(self.valid_path, 'r') as f:
            lines = f.readlines()
        self.valid_list = [line.split() for line in lines]
        print(len(self.valid_list))
        
        with open(self.test_path, 'r') as f:
            lines = f.readlines()
        self.test_list = [line.split() for line in lines]
        print(len(self.test_list))
    
    def counter_filter(self, raw_dataset, count=1):
        counter = collections.Counter([tk for tk in raw_dataset])
        counter = dict(filter(lambda x: x[1] >= count, counter.items()))
        return counter

    def preprocess(self):
        all_list = [*self.train_list,*self.valid_list,*self.test_list]
        entity_list = []
        relation_set = []
        for triple in all_list:
            entity_list.append(triple[0])
            entity_list.append(triple[2])
            relation_set.append(triple[1])
        entity_counter = self.counter_filter(entity_list, 5)
        relation_counter = self.counter_filter(relation_set, 1)
        for (i, entity) in enumerate(entity_counter.keys()):
            self.entity_map[entity] = i
        for (i, relation) in enumerate(relation_counter.keys()):
            self.relation_map[relation] = i
        self.entity_size = len(self.entity_map.keys())
        self.relation_size = len(self.relation_map.keys())

        self.train_triple = [(self.entity_map[i[0]], self.relation_map[i[1]], self.entity_map[i[0]]) 
                                for i in self.train_list
                                if (i[0] in self.entity_map.keys() and i[1] in self.relation_map.keys() and
                                    i[2] in self.entity_map.keys())]
        self.valid_triple = [(self.entity_map[i[0]], self.relation_map[i[1]], self.entity_map[i[0]]) 
                                for i in self.valid_list
                                if (i[0] in self.entity_map.keys() and i[1] in self.relation_map.keys() and
                                    i[2] in self.entity_map.keys())]
        self.test_triple = [(self.entity_map[i[0]], self.relation_map[i[1]], self.entity_map[i[0]]) 
                                for i in self.test_list
                                if (i[0] in self.entity_map.keys() and i[1] in self.relation_map.keys() and
                                    i[2] in self.entity_map.keys())]
        self.train_triple_size = len(self.train_triple)
        self.valid_triple_size = len(self.valid_triple)
        self.test_triple_size = len(self.test_triple)
        
        

if __name__ == "__main__":
    loader = DataLoader()
    loader.load_all()
    loader.preprocess()
