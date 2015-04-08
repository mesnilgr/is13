import numpy
import time
import sys
import subprocess
import os
import random

from is13.data import load
from is13.rnn.jordan import model
from is13.metrics.accuracy import conlleval
from is13.utils.tools import shuffle, minibatch, contextwin

if __name__ == '__main__':

    s = {'fold':3, # 5 folds 0,1,2,3,4
         'lr':0.0627142536696559,
         'verbose':1,
         'decay':False, # decay on the learning rate if improvement stops
         'win':7, # number of words in the context window
         'bs':9, # number of backprop through time steps
         'nhidden':100, # number of hidden units
         'seed':345,
         'emb_dimension':100, # dimension of word embedding
         'nepochs':50}

    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)

    # load the dataset
    train_set, valid_set, test_set, dic = load.atisfold(s['fold'])
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dic['words2idx'].iteritems())

    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex,  test_ne,  test_y  = test_set

    vocsize = len(set(reduce(\
                       lambda x, y: list(x)+list(y),\
                       train_lex+valid_lex+test_lex)))

    nclasses = len(set(reduce(\
                       lambda x, y: list(x)+list(y),\
                       train_y+test_y+valid_y)))
    
    nsentences = len(train_lex)

    # instanciate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = model(    nh = s['nhidden'],
                    nc = nclasses,
                    ne = vocsize,
                    de = s['emb_dimension'],
                    cs = s['win'] )

    # train with early stopping on validation set
    best_f1 = -numpy.inf
    s['clr'] = s['lr']
    for e in xrange(s['nepochs']):
        # shuffle
        shuffle([train_lex, train_ne, train_y], s['seed'])
        s['ce'] = e
        tic = time.time()
        for i in xrange(nsentences):
            cwords = contextwin(train_lex[i], s['win'])
            words  = map(lambda x: numpy.asarray(x).astype('int32'),\
                         minibatch(cwords, s['bs']))
            labels = train_y[i]

            for word_batch , label_last_word in zip(words, labels):
                rnn.train(word_batch, label_last_word, s['clr'])
                rnn.normalize()

            if s['verbose']:
                print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                sys.stdout.flush()
            
        # evaluation // back into the real world : idx -> words
        predictions_test = [ map(lambda x: idx2label[x], \
                             rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))\
                             for x in test_lex ]
        groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_y ]
        words_test = [ map(lambda x: idx2word[x], w) for w in test_lex]

        predictions_valid = [ map(lambda x: idx2label[x], \
                             rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))\
                             for x in valid_lex ]
        groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y ]
        words_valid = [ map(lambda x: idx2word[x], w) for w in valid_lex]

        # evaluation // compute the accuracy using conlleval.pl
        res_test  = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
        res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

        if res_valid['f1'] > best_f1:
            rnn.save(folder)
            best_f1 = res_valid['f1']
            if s['verbose']:
                print 'NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' '*20
            s['vf1'], s['vp'], s['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
            s['tf1'], s['tp'], s['tr'] = res_test['f1'],  res_test['p'],  res_test['r']
            s['be'] = e
            subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
            subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
        else:
            print ''
        
        # learning rate decay if no improvement in 10 epochs
        if s['decay'] and abs(s['be']-s['ce']) >= 10: s['clr'] *= 0.5 
        if s['clr'] < 1e-5: break

    print 'BEST RESULT: epoch', e, 'valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model', folder

