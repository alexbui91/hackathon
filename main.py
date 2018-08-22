"""
index of N/A in one_hot feature 0,41,73,112,129,146
39 in case of only seoul & not integrate china
"""

import tensorflow as tf
import numpy as np
import os
import sys
import time
import argparse

from model import Model
from engine import SparkEngine
import utils
import properties as p


def split_data(data, per):
    policies, claims, customers, labels = data
    lt = len(policies)
    train_lt = int(lt * per)
    policies, claims, customers, labels = np.asarray(policies, dtype=np.float32), np.asarray(claims, dtype=np.float32), np.asarray(customers, dtype=np.float32), np.asarray(labels, dtype=np.int32)
    r = np.random.permutation(lt)
    policies, claims, customers, labels = policies[r], claims[r], customers[r], labels[r]
    train_pol, train_cl, train_c, train_lb  = policies[:train_lt], claims[:train_lt], customers[:train_lt], labels[:train_lt]
    test_pol, test_cl, test_c, test_lb = policies[train_lt:], claims[train_lt:], customers[train_lt:], labels[train_lt:]
    return (train_pol, train_cl, train_c, train_lb), (test_pol, test_cl, test_c, test_lb)

def get_gpu_options(device="", gpu_devices="", gpu_fraction=None):
    gpu_options = None
    if "gpu" in device:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        os.environ["CUDA_VISIBLE_DEVICES"]= gpu_devices
    configs = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    return configs

def main(prefix="", url_feature="", url_weight=""):
    if not url_feature:
        engine = SparkEngine()
        data = engine.process_vectors()
        train, test = split_data(data, 0.8)
        utils.save_file("data/data.bin", (train, test))
    else:
        data = utils.load_file(url_feature)
        train, test = data
    train, valid = split_data(train, 0.8)
    model = Model()
    with tf.device('/cpu'):
        model.init_ops()
        print('==> initializing variables')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    tconfig = get_gpu_options()
 
    with tf.Session(config=tconfig) as session:
        sum_dir = 'summaries/train_' + time.strftime("%Y_%m_%d_%H_%M")
        train_writer = tf.summary.FileWriter(sum_dir, session.graph)
        session.run(init)
        best_val_epoch = 0
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        best_overall_val_loss = float('inf')

        if url_weight:
            print('==> restoring weights')
            saver.restore(session, '%s' % url_weight)

        print('==> starting training')
        val_losses, val_acces, best_preds, best_lb = [], [], [], []
        for i in xrange(p.total_iteration):
            print('Epoch {}'.format(i))
            start = time.time()
            train_loss, _ = model.run_epochs(train, session, i * p.total_iteration, train_writer)
            print('Training loss: {}'.format(train_loss))
            valid_loss, _ = model.run_epochs(valid, session,  i * p.total_iteration, train_writer, train=False)
            print('Validation loss: {}'.format(valid_loss))

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = i
                print('Saving weights')
                saver.save(session, 'weights/%s.weights' % prefix)

            if (i - best_val_epoch) > p.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))
        
        model.batch_size = 1
        test_loss, _ = model.run_epochs(valid, session,  i * p.total_iteration, train_writer, train=False)
        print('Validation loss: {}'.format(valid_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prefix")
    parser.add_argument("-f", "--data_path")

    args = parser.parse_args()

    main(args.prefix, args.data_path)