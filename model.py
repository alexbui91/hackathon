from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as numpy
from sklearn.metrics import f1_score
import sys

class Model():


    def __init__(self, hidden_layer_size=128, batch_size=32, learning_rate=0.01):
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.policy_vs = 53
        self.claim_vs = 15
        self.customer_vs = 22
        self.claim_length = 5
        self.customer_length = 5

    def init_ops(self):
        self.add_placeholders()
        self.inference()

    def add_placeholders(self):
        self.policy = tf.placeholder(tf.float32, shape=(self.batch_size, self.policy_vs))
        self.claim = tf.placeholder(tf.float32, shape=(self.batch_size, self.claim_length, self.claim_vs))
        self.customer = tf.placeholder(tf.float32, shape=(self.batch_size, self.customer_length, self.customer_vs))

        self.pred_labels = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.dropout_placeholder = tf.Variable(0.9, False, name="dropout", dtype=tf.float32)

    def inference(self):
        with tf.variable_scope("inference", initializer=self.initializer, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("claim_attention", initializer=self.initializer, reuse=tf.AUTO_REUSE):
                cl_a = self.get_attention(self.claim)

            with tf.variable_scope("customer_attention", initializer=self.initializer, reuse=tf.AUTO_REUSE):
                cus_a = self.get_attention(self.customer)

            inputs = tf.concat([self.policy, cl_a, cus_a], axis=1)
            hidden = tf.layers.dense(inputs, units=self.hidden_layer_size, name="policy_hidden", activation=tf.nn.relu)
            output = tf.squeeze(tf.layers.dense(inputs, units=1, name="output_hidden", activation=None))
            l = tf.losses.sigmoid_cross_entropy(self.pred_labels, output)
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.losses = tf.reduce_mean(l)
            gd = opt.compute_gradients(self.losses)
            self.train_op = opt.apply_gradients(gd)
            self.preds = tf.nn.softmax(output)

    def get_attention(self, inputs):
        # batch_size x length x 128
        cl_ = tf.layers.dense(inputs, units=self.hidden_layer_size, name="hidden_layer", activation=tf.nn.relu)
        # batch_size x self.claim_length
        cl_s_ = tf.squeeze(tf.layers.dense(cl_, units=1, name="hidden_softmax", activation=None))
        cl_s_ = tf.nn.softmax(cl_s_)
        cl_a = tf.multiply(tf.transpose(cl_, [2, 0, 1]), cl_s_)
        cl_a = tf.reduce_sum(tf.transpose(cl_a, [1, 2, 0]), axis=1)
        return cl_a

    def get_pred_rec(self, preds, labels):
        p =  
        r = 
        return p, r

    def run_epochs(self, data, session, num_epoch=0, train_writer=None, train=True):
        if not train:
            train_op = tf.no_op()
        else:
            train_op = self.train_op
        
        dt_length = len(data[0])
        total_steps = dt_length // self.batch_size
        total_loss = 0.0
        preds = []
        policy, claim, customer, labels = data

        for step in xrange(total_steps):
            index = range(step * self.batch_size, (step + 1) * self.batch_size)
            pt = policy[index]
            ct = customer[index]
            clt = claim[index]
            lt = labels[index]
            feed = {
                self.policy: pt,
                self.claim: clt,
                self.customer: ct,
                self.pred_labels: lt
            }

            loss, pred, _= session.run([self.losses, self.preds, train_op], feed_dict=feed)

            total_loss += loss
            if train:
                sys.stdout.write('\r{} / {} loss = {}'.format(
                    step, total_steps, total_loss / (step + 1)))
                sys.stdout.flush()
            else:
                preds += pred
        
        if train:
            sys.stdout.write("\r")
        if train_writer:
            if total_steps:
                summary = tf.Summary()
                if train:
                    name = "Train loss"
                else:
                    name = "Valid loss"
                total_loss = total_loss / total_steps
                summary.value.add(tag=name, simple_value=(total_loss / total_steps))
                train_writer.add_summary(summary, num_epoch)
        end = total_steps * self.batch_size
        labels = labels[:end]
        score = f1_score(labels, preds)
        return total_loss, score, preds
                
