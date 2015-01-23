import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class model(object):

    def __init__(self, nh, nc, ne, de, cs):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        '''
        #assert st in ['proba', 'argmax']

        self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Ws  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nc, nh)).astype(theano.config.floatX))
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.s0  = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))

        # bundle
        self.params = [ self.emb, self.Wx, self.Ws, self.W, self.bh, self.b, self.s0 ]
        self.names  = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 's0']
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y    = T.iscalar('y') # label

        def recurrence(x_t, s_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + \
                                 T.dot(s_tm1, self.Ws) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)[0]
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[None, self.s0], \
            n_steps=x.shape[0])

        p_y_given_x_lastword = s[-1,:]
        p_y_given_x_sentence = s
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.mean(T.log(p_y_given_x_lastword)[y])
        gradients = T.grad( nll, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))

        # theano functions
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.train = theano.function( inputs  = [idxs, y, lr],
                                      outputs = nll,
                                      updates = updates )

        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
