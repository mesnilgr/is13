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

def download_dropbox(fold):
    '''
    download from drop box in the meantime
    '''
    try:
        assert fold in [1, 3]
    except AssertionError:
        print('only the fold 1 and 3 are available since the UdeM deleted my server account without notice. apologies.')

    if fold == 1:
        os.system('wget -O atis.fold1.pkl.gz https://www.dropbox.com/s/81ar3d2ck3yavic/atis.fold1.pkl.gz?dl=0')
    elif fold == 3:
        os.system('wget -O atis.fold3.pkl.gz https://www.dropbox.com/s/tinjzm1b22tozn8/atis.fold3.pkl.gz?dl=0')


def load_dropbox(filename, fold):
    if not isfile(filename):
        #download('http://www-etud.iro.umontreal.ca/~mesnilgr/atis/'+filename)
        download_dropbox(fold)
    f = gzip.open(filename,'rb')
    return f

def load_udem(filename):
    if not isfile(filename):
        download('http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/'+filename)
    f = gzip.open(filename,'rb')
    return f


def atisfull():
    f = load_dropbox(PREFIX + 'atis.pkl')
    train_set, test_set, dicts = cPickle.load(f)
    return train_set, test_set, dicts

def atisfold(fold):
    assert fold in range(5)
    f = load_dropbox(PREFIX + 'atis.fold'+str(fold)+'.pkl.gz', fold)
    train_set, valid_set, test_set, dicts = cPickle.load(f)
    return train_set, valid_set, test_set, dicts

if __name__ == '__main__':

    ''' visualize a few sentences '''

    import pdb
    pdb.set_trace()
    w2ne, w2la = {}, {}
    train, _, test, dic = atisfold(3)

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
