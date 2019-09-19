from Model.TransE import *
from utils.Log import Log

if __name__ == '__main__':
    logger = Log(20, 'pg', False)

    ctx = gpu(1)
    param_path = './param/'
    dataset = 'WN18'
    model_name = 'WN18'
    mode = 'complex'
    sample_raw_negative = True
    useExistedModel = False
    autoEvaluate = True
    sparse = True
    margin = 2
    epoch_num = 1
    entity_dim = 20
    relation_dim = 20
    batch_size = 15000
    optimizer = 'sgd'
    lr = {'learning_rate': 0.1}
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
    train_triple_size = loader.train_triple_size
    total_batch_num = math.ceil(train_triple_size/batch_size)
    train_data = loader.train_triple
    valid_data = loader.valid_triple
    test_data = loader.test_triple
    print('Loading completed')
    
    net = TransE(entity_size, relation_size,
                        entity_dim, relation_dim,
                        sample_raw_negative=sample_raw_negative,
                        margin=margin,
                        ctx=ctx, logger=logger, sparse=sparse, param_path=param_path)
 
    net.load_relation_data(train_data, mode=mode, type='train')
    net.load_relation_data(valid_data, mode=mode, type='valid')
    if useExistedModel:
        print('Loading embeddings')
        net.load_embeddings(model_name=model_name)
    else:
        print('Initializing embeddings...')
        net.initialize(force_reinit=True, ctx=ctx)
        net.normalize_relation_parameters(ord=2)
        if autograd.is_recording():
            print('Relation normalization exposed.')
    print('Setting up trainer...')
    trainer = gluon.Trainer(net.collect_params(), optimizer, lr)


    all_start = time.time()
    print('Start iteration:')

    old_model_epoch = len(net.get_old_log())
    for epoch in range(epoch_num):
        epoch_loss = 0
        epoch_start = time.time()
        net.normalize_entity_parameters(ord=2)
        if autograd.is_recording():
            print('Entity normalization exposed.')
        checkpoint = time.time()
        logger.debug('Entity normalization completed, time used: {}'.format(str(checkpoint-epoch_start)))
        # logger.info('*'*40)
        for current_batch_num in range(total_batch_num):
            # print('current batch: {}'.format(str(current_batch_num)))
            t1 = time.time()
            with autograd.record():
                output = net(current_batch_num*batch_size, 
                            current_batch_num*batch_size+batch_size)
                total_loss = output.sum()
                output = output.mean()
                t2 = time.time()
            t3 = time.time()
            output.backward()
            t4 = time.time()
            logger.debug('Forward time: {}'.format(str(t2-t1)))
            logger.debug('Calc loss time: {}'.format(str(t3-t2)))
            logger.debug('Backward time: {}'.format(str(t4-t3)))

            # print('epoch {} batch {}/{} completed, time used: {}, epoch time remaining: {}'.format(
            #         str(epoch),
            #         str(current_batch_num),
            #         str(total_batch_num),
            #         str(time.time()-epoch_start),
            #         str((time.time()-epoch_start)/(current_batch_num+1)*(total_batch_num-current_batch_num-1))))
            
            checkpoint = time.time()
            trainer.step(1)
            logger.debug('Back-propagation time: {}'.format(str(time.time()-checkpoint)))
            checkpoint = time.time()
            batch_total_loss = total_loss.asscalar()
            batch_info = 'epoch {} batch {} time: {} total loss: {}, avg loss: {}'.format(
                    str(epoch),
                    str(epoch+old_model_epoch), 
                    str(time.time() - t1),
                    str(batch_total_loss),
                    str(batch_total_loss/batch_size))
            # logger.info(batch_info)
            epoch_loss += batch_total_loss
        #TODO
        epoch_time = time.time() - epoch_start
        epoch_info = 'epoch {} time: {} \n\t total loss: {},\n\t avg loss: {}'.format(
                str(epoch+old_model_epoch), 
                str(epoch_time),
                str(epoch_loss),
                str(epoch_loss / train_triple_size))
        logger.info(epoch_info)
        net.add_training_log(epoch+old_model_epoch, epoch_loss, epoch_time)
        if epoch==0 or (epoch+1) % 50 == 0:
            draw(net.get_loss_trend(), False, model_name+'.png')
            print('Auto-save parameters.')
            print(epoch_info)
            net.save_embeddings(model_name=model_name)
            checkpoint = time.time()
            if autoEvaluate and ((epoch+1) % 100 == 0 or epoch == 0):
                (total, hit) = net.evaluate(mode='valid', k=k, ord=1)
                for i in range(len(k)):
                    logger.info('Evaluation time: {} Hit@ {}: {}'.format(str(time.time()-checkpoint), str(k[i]), str(hit[i]*1.0/total)))
                    print('Evaluation time: {} Hit@ {}: {} {}/{}'.format(str(time.time()-checkpoint), str(k[i]), str(hit[i]*1.0/total), str(int(hit[i])),  str(int(total))))
                # if hit*1.0/total > 0.5:
                #     break
                # TODO
            else:
                print('Skip Evaluation.')

    net.save_embeddings(model_name=model_name)
    # net.dump('relations.txt', loss)
    draw(net.get_loss_trend(), model_name+'.png')
