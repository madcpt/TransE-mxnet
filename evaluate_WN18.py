from Model.TransE import *
from utils.Log import Log

if __name__ == '__main__':
    # logger = Log(20, 'pg', False)

    ctx = gpu(1)
    param_path = './param/'
    dataset = 'WN18'
    model_name = 'WN18'
    mode = 'complex'
    sparse = True
    margin = 5
    entity_dim = 50
    relation_dim = 50
    k = [1, 10, 20, 50, 100, 1000, 40000]

    loader = DataLoader(dataset)
    print('Start loading data from {}'.format(loader.train_path))
    print('Start loading data from {}'.format(loader.valid_path))
    print('Start loading data from {}'.format(loader.test_path))
    loader.load_all()
    print('Start preprocessing...')
    loader.preprocess(filter_occurance=1)
    entity_size = loader.entity_size
    relation_size = loader.relation_size
    test_data = loader.test_triple
    print('Loading completed')
    
    net = TransE(entity_size, relation_size,
                        entity_dim, relation_dim,
                        margin=margin,
                        ctx=ctx, logger=None, sparse=sparse, param_path=param_path)

    print('Loading model...')
    net.load_relation_data(test_data, mode=mode, type='test')
    net.load_embeddings(model_name=model_name)
    print('Start evaluating')
    checkpoint = time.time()
    (total, hit) = net.evaluate(k=k)
    for i in range(len(k)):
        print('Evaluation time: {} Hit@ {}: {} {}/{}'.format(str(time.time()-checkpoint), str(k[i]), str(hit[i]*1.0/total), str(int(hit[i])),  str(int(total))))

    (total, hit) = net.evaluate(k=k, evaluation_type='head_prediction')
    for i in range(len(k)):
        print('Evaluation time: {} Hit@ {}: {} {}/{}'.format(str(time.time()-checkpoint), str(k[i]), str(hit[i]*1.0/total), str(int(hit[i])),  str(int(total))))
    # TODO
