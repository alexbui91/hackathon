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


def array_to_str(policies, preds, delimiter="\n"):
    tmp = ""
    l = len(preds)
    e = l - 1
    for i in xrange(l):
        tmp += "%i, %i" % (policies[i], preds[i])
        if i < e:
            tmp += delimiter
    return tmp


def save_file(name, obj):
    with open(name, 'wb') as f:
        f.write(obj)


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
    if "gpu" in device:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        os.environ["CUDA_VISIBLE_DEVICES"]= gpu_devices
    configs = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    return configs

def main(prefix="zhongan", url_feature="", url_weight="", policy_one_hot=True):
    if not url_feature:
        engine = SparkEngine()
        train_data = engine.get_train_data()
        utils.save_file("data/train_data_%s.bin" % prefix, train_data)
    else:
        train_data = utils.load_file(url_feature)
    train, valid = split_data(train_data, 0.8)
    model = Model()
    with tf.device('/gpu:3'):
        model.init_ops()
        print('==> initializing variables')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    tconfig = get_gpu_options("gpu", "3,4")
 
    with tf.Session(config=tconfig) as session:
        sum_dir = 'summaries/train_' + time.strftime("%Y_%m_%d_%H_%M")
        train_writer = tf.summary.FileWriter(sum_dir, session.graph)
        session.run(init)
        best_val_epoch = 0
        best_val_loss = float('inf')

        if url_weight:
            print('==> restoring weights')
            saver.restore(session, '%s' % url_weight)

        print('==> starting training')
        for i in xrange(p.total_iteration):
            print('Epoch {}'.format(i))
            start = time.time()
            train_loss, f1_score, _ = model.run_epochs(train, session, i * p.total_iteration, train_writer)
            print('Training loss: {} | f1_score {}'.format(train_loss, f1_score))
            valid_loss, vf1_score, _ = model.run_epochs(valid, session,  i * p.total_iteration, train_writer, train=False)
            print('Validation loss: {} | f1_score {}'.format(valid_loss, vf1_score))

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = i
                print('Saving weights')
                saver.save(session, 'weights/%s.weights' % prefix)

            if (i - best_val_epoch) > p.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))
        
        # model.batch_size = 1
        # _, tf1_score, preds = model.run_epochs(test_data, session,  i * p.total_iteration, train_writer, train=False)
        # print('f1_score: {}'.format(tf1_score))
        # tmp = array_to_str(preds)
        # save_file("prediction.txt", tmp)


def test(prefix="zhongan", url_feature="", url_weight="", policy_one_hot=True):
    if not url_feature:
        engine = SparkEngine()
        test_data = engine.get_test_data()
        utils.save_file("data/test_data_%s.bin" % prefix, test_data)
    else:
        test_data = utils.load_file(url_feature)
    
    policies, claims, customers, _, policy_ids = test_data
    policies, claims, customers, policy_ids = np.asarray(policies, dtype=np.float32), np.asarray(claims, dtype=np.float32), np.asarray(customers, dtype=np.float32), np.asarray(policy_ids, dtype=np.int32)
    test_data = (policies, claims, customers, [])
    model = Model(batch_size=1, is_test=True)
    with tf.device('/gpu:3'):
        model.init_ops()
        print('==> initializing variables')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    tconfig = get_gpu_options("gpu", "3,4")
 
    with tf.Session(config=tconfig) as session:
        session.run(init)
        if url_weight:
            print('==> restoring weights')
            saver.restore(session, '%s' % url_weight)
        
        _, _, preds = model.run_epochs(test_data, session, 0, None, train=False)
        tmp = array_to_str(policy_ids, preds)
        save_file("prediction_%s.txt" % prefix, tmp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prefix", default="zhongan")
    parser.add_argument("-f", "--data_path")
    parser.add_argument("-w", "--weight")
    parser.add_argument("-t", "--test", type=int, default=0)
    parser.add_argument("-o", "--one_hot_policy", type=int, default=1)

    args = parser.parse_args()
    if args.test:
        test(args.prefix, args.data_path, args.weight, args.one_hot_policy)
    else:
        main(args.prefix, args.data_path, args.weight, args.one_hot_policy)