from Model import *
from utils.Log import Log

if __name__ == '__main__':
    logger = Log(10, 'pg', False)

    local = False

    if local:
        ctx = gpu(0)
        param_path = './param/'
        model_name = 'simple'
        mode = 'simple'
        isTrain = True
        sample_raw_negative = False
        isContinueTrain = False
        sparse = False
        margin = 0.3 
        epoch_num = 3000
        k = 1
    else:
        ctx = gpu(0)
        param_path = './param/'
        model_name = 'WN'
        mode = 'complex'
        isTrain = True
        sample_raw_negative = True
        isContinueTrain = False
        autoEvaluate = False
        sparse = True
        margin=10
        epoch_num = 1
        k = 20

    if mode ==  'simple':
        entity_size = 5
        relation_size = 2
        entity_dim = 2
        relation_dim = 2
        batch_size = 3
        train_data = build_simple_dataset(entity_dim=entity_dim, 
                                relation_dim=relation_dim, ctx=ctx)
        train_triple_size = len(train_data)
        valid_data = train_data
        valid_triple_size = train_triple_size
        test_data = build_simple_dataset(entity_dim=entity_dim, 
                                relation_dim=relation_dim, ctx=ctx)
        test_triple_size = len(test_data)
        total_batch_num = math.ceil(train_triple_size/batch_size)
    elif mode == 'complex':
        loader = DataLoader()
        print('Start loading data from {}'.format(loader.train_path))
        print('Start loading data from {}'.format(loader.valid_path))
        print('Start loading data from {}'.format(loader.test_path))
        loader.load_all()
        print('Start preprocessing...')
        loader.preprocess(filter_occurance=1)
        entity_size = loader.entity_size
        relation_size = loader.relation_size
        train_triple_size = loader.train_triple_size
        entity_dim = 50
        relation_dim = 50
        batch_size = 150000
        total_batch_num = math.ceil(train_triple_size/batch_size)
        train_data = loader.train_triple
        valid_data = loader.valid_triple
        test_data = loader.test_triple
        print('Loading completed')
    
    net = TransE(entity_size, relation_size,
                        entity_dim, relation_dim,
                        # negative_sampling_rate=0.3, 
                        sample_raw_negative=sample_raw_negative,
                        margin=margin,
                        ctx=ctx, logger=logger, sparse=sparse, param_path=param_path)
    loss = gloss.L1Loss()

    if isTrain:
        net.load_relation_data(train_data, mode=mode, type='train')
        net.load_relation_data(valid_data, mode=mode, type='valid')
        if isContinueTrain:
            print('Loading embeddings')
            net.load_embeddings(model_name=model_name)
        else:
            print('Initializing embeddings...')
            net.initialize(force_reinit=True, ctx=ctx)
            net.normalize_relation_parameters()
        print('Setting up trainer...')
        trainer = gluon.Trainer(net.collect_params(), 'sgd',
                                {'learning_rate': 0.1})


        p = []
        all_start = time.time()
        print('Start iteration:')

        for epoch in range(epoch_num):
            epoch_loss = 0
            epoch_start = time.time()
            net.normalize_entity_parameters()
            checkpoint = time.time()
            logger.info('Entity normalization completed, time used: {}'.format(str(checkpoint-epoch_start)))
            # logger.info('*'*40)
            for current_batch_num in range(total_batch_num):
                # print('current batch: {}'.format(str(current_batch_num)))
                with autograd.record():
                    t1 = time.time()
                    output = net(current_batch_num*batch_size, 
                                current_batch_num*batch_size+batch_size)
                    print(output)
                    t2 = time.time()
                    l = loss(output, nd.zeros(output.shape, ctx=ctx))
                    t3 = time.time()
                batch_total_loss = l.sum().asscalar()
                    # logger.debug(l)
                    # l = l.mean()
                logger.info('Forward time: {}'.format(str(t2-t1)))
                logger.info('Calc loss time: {}'.format(str(t3-t2)))

                # print('epoch {} batch {}/{} completed, time used: {}, epoch time remaining: {}'.format(
                #         str(epoch),
                #         str(current_batch_num),
                #         str(total_batch_num),
                #         str(time.time()-epoch_start),
                #         str((time.time()-epoch_start)/(current_batch_num+1)*(total_batch_num-current_batch_num-1))))
                
                checkpoint = time.time()
                l.backward()
                logger.info('Backward time: {}'.format(str(time.time()-checkpoint)))
                checkpoint = time.time()
                trainer.step(batch_size)
                # print('step time: {}'.format(str(time.time()-checkpoint)))
                checkpoint = time.time()
                batch_info = 'epoch {} batch {} time: {} \n\t total loss: {},\n\t avg loss: {}'.format(
                        str(epoch),
                        str(current_batch_num), 
                        str(time.time() - t1),
                        str(batch_total_loss),
                        str(batch_total_loss/batch_size))
                # logger.info(batch_info)
                # print(batch_info)
                epoch_loss += batch_total_loss
            p.append(epoch_loss/len(train_data)) 
            #TODO
            epoch_info = 'epoch {} time: {} \n\t total loss: {},\n\t avg loss: {}'.format(
                    str(epoch), 
                    str(time.time() - epoch_start),
                    str(epoch_loss),
                    str(epoch_loss / train_triple_size))
            logger.info(epoch_info)
            if (epoch+1) % 100 == 0:
                print('Auto-save parameters.')
                print(epoch_info)
                net.save_embeddings(model_name=model_name)
                checkpoint = time.time()
                if autoEvaluate:
                    (total, hit) = net.evaluate(mode='valid', k=k)
                    net.logger.debug('Evaluation time: {} accuracy: {}'.format(str(time.time()-checkpoint), str(hit*1.0/total)))
                    print('Evaluation time: {} accuracy: {}'.format(str(time.time()-checkpoint), str(hit*1.0/total)))
                    if hit*1.0/total > 0.5:
                        break
                else:
                    print('Skip Evaluation.')


            # net.save_parameters('{}model.params'.format(param_path))
        net.save_embeddings(model_name=model_name)
        # net.dump('relations.txt', loss)
        draw(p)
    else:
        net.load_embeddings(model_name=model_name)
        net.load_relation_data(test_data, mode=mode, type='test')
        print('Initializing model...')
        net.initialize(force_reinit=True, ctx=ctx)
        (total, hit) = net.evaluate(k=100)
        print(total)
        print(hit)
        print(hit/total)
        # TODO

    # for i in range(5):
    #     print(net.predict_with_h_r(i,0))
    # for i in range(5):
    #     print(net.predict_with_h_r(i,1))
