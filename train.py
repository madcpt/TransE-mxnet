
logger = Log(10, 'pg', False)

ctx = gpu(0)
mode = 'simple'
param_path = './param/'

if mode ==  'simple':
    entity_size = 5
    relation_size = 2
    entity_dim = 3
    relation_dim = 3
    batch_size = 2
    train_data = build_simple_dataset(entity_dim=entity_dim, 
                            relation_dim=relation_dim, ctx=ctx)
    train_triple_size = len(train_data)
    total_batch_num = train_triple_size //  batch_size + 1
elif mode == 'complex':
    loader = DataLoader()
    print('Start loading data from {}'.format(loader.train_path))
    print('Start loading data from {}'.format(loader.valid_path))
    print('Start loading data from {}'.format(loader.test_path))
    loader.load_all()
    print('Start preprocessing...')
    loader.preprocess()
    entity_size = loader.entity_size
    relation_size = loader.relation_size
    train_triple_size = loader.train_triple_size
    entity_dim = 50
    relation_dim = 50
    batch_size = 500
    total_batch_num = train_triple_size // batch_size + 1
    train_data = loader.train_triple
    print('Loading completed')

net = FancyMLP(entity_size, relation_size,
                    entity_dim, relation_dim,
                    # negative_sampling_rate=0.3, 
                    margin=2,
                    ctx=ctx, logger=logger)

net.load_relation_data(train_data, mode=mode, type='train')
print('Initializing model...')
net.initialize(force_reinit=True, ctx=ctx)

print('Setting up trainer...')
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.01})

net.normalize_relation_parameters()

loss = gloss.L2Loss()

p = []

print('Start iteration:')

all_start = time.time()
for epoch in range(10):
    epoch_loss = 0
    epoch_start = time.time()
    net.normalize_entity_parameters()
    checkpoint = time.time()
    print('normalization completed, time used: {}'.format(str(checkpoint-epoch_start)))
    # logger.info('*'*40)
    for current_batch_num in range(total_batch_num):
        print('current batch: {}'.format(str(current_batch_num)))
        with autograd.record():
            t1 = time.time()
            output = net(current_batch_num*batch_size, 
                        current_batch_num*batch_size+batch_size)
            # logger.debug(output)
            t2 = time.time()
            l = loss(output, nd.zeros(output.shape, ctx=ctx))
            t3 = time.time()
            batch_total_loss = l.sum().asscalar()
            # logger.debug(l)
            # l = l.mean()
        # print(t2-t1)
        # print(t3-t2)
        print('epoch {} batch {}/{} completed, time used: {}, epoch time remaining: {}'.format(
                str(epoch),
                str(current_batch_num),
                str(total_batch_num),
                str(time.time()-epoch_start),
                str((time.time()-epoch_start)/(current_batch_num+1)*total_batch_num)))
        checkpoint = time.time()
        l.backward()
        # print('backward time: {}'.format(str(time.time()-checkpoint)))
        checkpoint = time.time()
        trainer.step(batch_size)
        # print('step time: {}'.format(str(time.time()-checkpoint)))
        checkpoint = time.time()
        batch_info = 'epoch {} batch {} time: {} \n\t total loss: {},\n\t avg loss: {}'.format(
                str(epoch),
                str(current_batch_num), 
                str(time.time() - checkpoint),
                str(batch_total_loss),
                str(batch_total_loss/batch_size))
        logger.info(batch_info)
        # print(batch_info)
        epoch_loss += batch_total_loss
    p.append(epoch_loss/total_batch_num) 
    #TODO
    epoch_info = 'epoch {} time: {} \n\t total loss: {},\n\t avg loss: {}'.format(
            str(epoch), 
            str(time.time() - checkpoint),
            str(epoch_loss),
            str(epoch_loss / train_triple_size))
    logger.info(epoch_info)
    print(epoch_info)
    net.save_parameters('{}model.params'.format(param_path))
net.dump('relations.txt', loss)
draw(p)

# for i in range(5):
#     print(net.predict_with_h_r(i,0))
# for i in range(5):
#     print(net.predict_with_h_r(i,1))
