import gzip
import cPickle
import urllib
import os
import random

from os.path import isfile

PREFIX = os.getenv('ATISDATA', '')

def download(origin):
    '''
    download the corresponding atis file
    from http://www-etud.iro.umontreal.ca/~mesnilgr/atis/
    '''
    print 'Downloading data from %s' % origin
    name = origin.split('/')[-1]
    urllib.urlretrieve(origin, name)

def download_dropbox():
    ''' 
    download from drop box in the meantime
    '''
    print 'Downloading data from https://www.dropbox.com/s/3lxl9jsbw0j7h8a/atis.pkl?dl=0'
    os.system('wget -O atis.pkl https://www.dropbox.com/s/3lxl9jsbw0j7h8a/atis.pkl?dl=0')

def load(filename):
    if not isfile(filename):
        #download('http://www-etud.iro.umontreal.ca/~mesnilgr/atis/'+filename)
        download_dropbox()
    #f = gzip.open(filename,'rb')
    f = open(filename,'rb')
    return f

def atisfull():
    #f = load(PREFIX + 'atis.pkl.gz')
    f = load(PREFIX + 'atis.pkl')
    train_set, test_set, dicts = cPickle.load(f)
    return train_set, test_set, dicts

def atisfold(fold):
    assert fold in range(5)
    #f = load(PREFIX + 'atis.fold'+str(fold)+'.pkl.gz')
    f = load(PREFIX + 'atis.pkl')
    train_set, test_set, dicts = cPickle.load(f)
    new_train_set, new_valid_set = [], []
    for subset in train_set:
        random.seed(fold)
        random.shuffle(subset)
        new_train_set += [subset[:978]]
        new_valid_set += [subset[-978:]]
    return new_train_set, new_valid_set, test_set, dicts
 
if __name__ == '__main__':
    
    ''' visualize a few sentences '''

    import pdb
    data = atisfull()

    w2ne, w2la = {}, {}
    train, test, dic = data
    
    w2idx, ne2idx, labels2idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']
    
    idx2w  = dict((v,k) for k,v in w2idx.iteritems())
    idx2ne = dict((v,k) for k,v in ne2idx.iteritems())
    idx2la = dict((v,k) for k,v in labels2idx.iteritems())

    test_x,  test_ne,  test_label  = test
    train_x, train_ne, train_label = train
    wlength = 35

    for e in ['train','test']:
      for sw, se, sl in zip(eval(e+'_x'), eval(e+'_ne'), eval(e+'_label')):
        print 'WORD'.rjust(wlength), 'LABEL'.rjust(wlength)
        for wx, la in zip(sw, sl): print idx2w[wx].rjust(wlength), idx2la[la].rjust(wlength)
        print '\n'+'**'*30+'\n'
        pdb.set_trace()
