#!/usr/bin/env python3
# -*- coding: utf-8
import sys
# some simple python commands
sys.path.append('/usr/local/lib/python3.6/site-packages')
sys.path.append('/content/DeepXML_XMLCNN/XML-CNN/utils')
sys.path.append('/content/DeepXML_XMLCNN/XML-CNN/code/models')
sys.path.append('/content/DeepXML_XMLCNN/XML-CNN/code/')
sys.path.append('/content/DeepXML_XMLCNN/XML-CNN/utils')
sys.path.append('/content/DeepXML_XMLCNN/deepXML')

import os
import click
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from logzero import logger
from header import *
from cnn_train import *
from cnn_test import *
import pdb
from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import get_data, get_mlb, get_word_emb, output_res
from deepxml.models import Model
from deepxml.tree import FastAttentionXML
from deepxml.networks import AttentionRNN


def load_cnn_data():
  # ------------------------ Params -------------------------------------------------------------------------------
  parser = argparse.ArgumentParser(description='Process some integers.')

  parser.add_argument('--zd', dest='Z_dim', type=int, default=100, help='Latent layer dimension')
  parser.add_argument('--mb', dest='mb_size', type=int, default=20, help='Size of minibatch, changing might result in latent layer variance overflow')
  # parser.add_argument('--hd', dest='h_dim', type=int, default=600, help='hidden layer dimension')
  parser.add_argument('--lr', dest='lr', type=int, default=1e-2, help='Learning Rate')
  parser.add_argument('--p', dest='plot_flg', type=int, default=0, help='1 to plot, 0 to not plot')
  parser.add_argument('--e', dest='num_epochs', type=int, default=100, help='step for displaying loss')

  parser.add_argument('--d', dest='disp_flg', type=int, default=0, help='display graphs')
  parser.add_argument('--sve', dest='save', type=int, default=1, help='save models or not')
  parser.add_argument('--ss', dest='save_step', type=int, default=10, help='gap between model saves')
  parser.add_argument('--mn', dest='model_name', type=str, default='', help='model name')
  parser.add_argument('--tr', dest='training', type=int, default=1, help='model name')
  parser.add_argument('--lm', dest='load_model', type=str, default="", help='model name')
  parser.add_argument('--ds', dest='data_set', type=str, default="rcv", help='dataset name')

  parser.add_argument('--pp', dest='pp_flg', type=int, default=0, help='1 is for min-max pp, 2 is for gaussian pp, 0 for none')
  parser.add_argument('--loss', dest='loss_type', type=str, default="BCELoss", help='Loss')

  parser.add_argument('--hidden_dims', type=int, default=512, help='hidden layer dimension')
  parser.add_argument('--sequence_length',help='max sequence length of a document', type=int,default=500)
  parser.add_argument('--embedding_dim', help='dimension of word embedding representation', type=int, default=300)
  parser.add_argument('--model_variation', help='model variation: CNN-rand or CNN-pretrain', type=str, default='pretrain')
  parser.add_argument('--pretrain_type', help='pretrain model: GoogleNews or glove', type=str, default='glove')
  parser.add_argument('--vocab_size', help='size of vocabulary keeping the most frequent words', type=int, default=30000)
  parser.add_argument('--drop_prob', help='Dropout probability', type=int, default=.3)
  parser.add_argument('--load_data', help='Load Data or not', type=int, default=0)
  parser.add_argument('--mg', dest='multi_gpu', type=int, default=0, help='1 for 2 gpus and 0 for normal')
  parser.add_argument('--filter_sizes', help='number of filter sizes (could be a list of integer)', type=int, default=[2, 4, 8], nargs='+')
  parser.add_argument('--num_filters', help='number of filters (i.e. kernels) in CNN model', type=int, default=32)
  parser.add_argument('--pooling_units', help='number of pooling units in 1D pooling layer', type=int, default=32)
  parser.add_argument('--pooling_type', help='max or average', type=str, default='max')
  parser.add_argument('--model_type', help='glove or GoogleNews', type=str, default='glove')
  parser.add_argument('--num_features', help='50, 100, 200, 300', type=int, default=300)
  parser.add_argument('--dropouts', help='0 for not using, 1 for using', type=int, default=0)
  parser.add_argument('--clip', help='gradient clipping', type=float, default=1000)
  parser.add_argument('--dataset_gpu', help='load dataset in full to gpu', type=int, default=1)
  parser.add_argument('--dp', dest='dataparallel', help='to train on multiple GPUs or not', type=int, default=0)
  parser.add_argument('--data_conf', dest='data_conf', type=str, default="/content/DeepXML_XMLCNN/deepXML/configure/datasets/EUR-Lex.yaml", help='dataset conf file')
  parser.add_argument('--model_conf', dest="model_conf" , type=str, default="/content/DeepXML_XMLCNN/deepXML/configure/models/AttentionXML-EUR-Lex.yaml", help='model conf')
  parser.add_argument('--t', dest='treeid', type=int, default=0, help='treeid')
  parser.add_argument('--mode', dest="mode" , type=str, default="train", help='train eval')
  params = parser.parse_args()
  if(len(params.model_name)==0):
    params.model_name = "Gen_data_CNN_Z_dim-{}_mb_size-{}_hidden_dims-{}_preproc-{}_loss-{}_sequence_length-{}_embedding_dim-{}_params.vocab_size={}".format(params.Z_dim, params.mb_size, params.hidden_dims, params.pp_flg, params.loss_type, params.sequence_length, params.embedding_dim, params.vocab_size)

  print('Saving Model to: ' + params.model_name)

  # ------------------ data ----------------------------------------------
  # params.data_path = '../datasets/' + params.data_set
  params.data_path = '/content/drive/MyDrive/DeepXML/EUR-Lex'
  x_tr, x_te, y_tr, y_te, params.vocabulary, params.vocabulary_inv, params = save_load_data(params, save=params.load_data)

  params = update_params(params)
  # -----------------------  Loss ------------------------------------
  params.loss_fn = torch.nn.BCELoss(size_average=False)
  # -------------------------- Params ---------------------------------------------
  if params.model_variation=='pretrain':
      embedding_weights = load_word2vec(params)
  else:
      embedding_weights = None

  if torch.cuda.is_available():
      params.dtype = torch.cuda.FloatTensor
  else:
      params.dtype = torch.FloatTensor
  # embedding_weights = ""
  return x_tr, y_tr, x_te, y_te, embedding_weights, params


# @click.command()
# @click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
# @click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of model configure yaml.')
# @click.option('--mode', type=click.Choice(['train', 'eval']), default=None)
# @click.option('-t', '--tree-id', type=click.INT, default=None)

def main():

    x_tr, y_tr, x_te, y_te, embedding_weights, params = load_cnn_data()
    # print(type(x_tr))
    # print(x_tr.get_shape())
    # print("loaded")
    data_cnf = params.data_conf
    model_cnf = params.model_conf
    tree_id = params.treeid
    mode = params.mode
    print("data conf is ", data_cnf)
    print("model conf is ", model_cnf)
    tree_id = '-Tree-{tree_id}'.format(
        tree_id=tree_id) if tree_id is not None else ''
    print("tree_id....", tree_id)
    print("mode is ", mode)
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
    model_path = os.path.join(
        model_cnf['path'], '{}-{}{}'.format(model_name, data_name, tree_id))

    emb_init = get_word_emb(data_cnf['embedding']['emb_init'])
    logger.info('Model Name: {}'.format(model_name))

    if mode is None or mode == 'train':
        logger.info('Loading Training and Validation Set')
        train_x, train_labels = get_data(
            data_cnf['train']['texts'], data_cnf['train']['labels'])
        if 'size' in data_cnf['valid']:
            random_state = data_cnf['valid'].get('random_state', 1240)
            train_x, valid_x, train_labels, valid_labels = train_test_split(train_x, train_labels,
                                                                            test_size=data_cnf['valid']['size'],
                                                                            random_state=random_state)
        else:
            valid_x, valid_labels = get_data(
                data_cnf['valid']['texts'], data_cnf['valid']['labels'])
        mlb = get_mlb(data_cnf['labels_binarizer'],
                      np.hstack((train_labels, valid_labels)))
        train_y, valid_y = mlb.transform(
            train_labels), mlb.transform(valid_labels)
        labels_num = len(mlb.classes_)

        logger.info('Number of Labels: {}'.format(labels_num))
        logger.info('Size of Training Set: {}'.format(len(train_x)))
        logger.info('Size of Validation Set: {}'.format(len(valid_x)))

        logger.info('Training')
        print("here.......")

        if 'cluster' not in model_cnf:
            train_loader = DataLoader(MultiLabelDataset(train_x, train_y),
                                      model_cnf['train']['batch_size'], shuffle=True, num_workers=4)
            valid_loader = DataLoader(MultiLabelDataset(valid_x, valid_y, training=False),
                                      model_cnf['valid']['batch_size'], num_workers=4)
            model = Model(network=AttentionRNN, labels_num=labels_num, model_path=model_path, emb_init=emb_init,
                          **data_cnf['model'], **model_cnf['model'])
            model.train_xml_deep(train_loader, valid_loader,  x_tr, y_tr, x_te, y_te, embedding_weights, params, **model_cnf['train'])
    #     else:
    #         model = FastAttentionXML(labels_num, data_cnf, model_cnf, tree_id)
    #         model.train(train_x, train_y, valid_x, valid_y, mlb)
    #         logger.info(type(model))
    #         logger.info(type(model.models))
    #         logger.info(type(model.models[0]))
    #     logger.info('Finish Training')

    # if mode is None or mode == 'eval':
    #     logger.info('Loading Test Set')
    #     mlb = get_mlb(data_cnf['labels_binarizer'])
    #     labels_num = len(mlb.classes_)
    #     test_x, _ = get_data(data_cnf['test']['texts'], None)
    #     logger.info('Size of Test Set: {}'.format(len(test_x)))

    #     logger.info('Predicting')
    #     if 'cluster' not in model_cnf:
    #         test_loader = DataLoader(MultiLabelDataset(test_x), model_cnf['predict']['batch_size'],
    #                                  num_workers=4)
    #         if model is None:
    #             model = Model(network=AttentionRNN, labels_num=labels_num, model_path=model_path, emb_init=emb_init,
    #                           **data_cnf['model'], **model_cnf['model'])
    #         scores, labels = model.predict(
    #             test_loader, k=model_cnf['predict'].get('k', 100))
    #     else:
    #         if model is None:
    #             model = FastAttentionXML(
    #                 labels_num, data_cnf, model_cnf, tree_id)
    #         logger.info(type(model))
    #         logger.info(type(model.models))
    #         logger.info(type(model.models[0]))

    #         scores, labels = model.predict(test_x)

    #     logger.info('Finish Predicting')
    #     labels = mlb.classes_[labels]
    #     output_res(data_cnf['output']['res'],
    #                '{}-{}{}'.format(model_name, data_name, tree_id), scores, labels)


if __name__ == '__main__':

    main()
