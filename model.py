from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
import sys
import properties as pr

class Model():


    def __init__(self, hidden_layer_size=128, batch_size=64, learning_rate=0.01, is_test=False, net=1, loss_function="sigmoid"):
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.policy_vs = pr.policy_dims
        self.claim_vs = pr.cl_dims
        self.customer_vs = pr.c_dims
        self.claim_length = pr.max_claim
        self.customer_length = pr.max_customer
        self.is_test=is_test
        self.net = net
        self.loss_function = loss_function

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
            if self.net == 0:
                hidden = tf.layers.dense(inputs, units=self.hidden_layer_size, name="policy_hidden", activation=tf.nn.relu)
            elif self.net == 1:
                hidden = tf.layers.dense(inputs, units=self.hidden_layer_size, name="policy_hidden", activation=tf.nn.tanh)
            elif self.net == 2:
                hidden = tf.layers.dense(inputs, units=self.hidden_layer_size, name="policy_hidden_1", activation=tf.nn.tanh)
                hidden = tf.layers.dense(hidden, units=self.hidden_layer_size/2, name="policy_hidden_2", activation=tf.nn.tanh)
                hidden = tf.layers.dense(hidden, units=self.hidden_layer_size/4, name="policy_hidden_3", activation=tf.nn.relu)
            elif self.net == 3:
                hidden = tf.layers.dense(inputs, units=self.hidden_layer_size, name="policy_hidden_1", activation=tf.nn.tanh)
                hidden = tf.layers.dense(hidden, units=self.hidden_layer_size/4, name="policy_hidden_2", activation=tf.nn.tanh)
            elif self.net == 4:
                hidden = tf.layers.dense(inputs, units=self.hidden_layer_size, name="policy_hidden_1", activation=tf.nn.tanh)
                hidden = tf.layers.dense(hidden, units=self.hidden_layer_size/2, name="policy_hidden_2", activation=tf.nn.tanh)
                hidden = tf.layers.dense(hidden, units=self.hidden_layer_size/4, name="policy_hidden_3", activation=tf.nn.tanh)
            else:
                hidden = tf.layers.dense(inputs, units=self.hidden_layer_size, name="policy_hidden_1", activation=tf.nn.tanh)
                hidden = tf.layers.dense(hidden, units=self.hidden_layer_size/2, name="policy_hidden_2", activation=tf.nn.tanh)

            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            if self.loss_function == "softmax":
                output = tf.squeeze(tf.layers.dense(hidden, units=2, name="output_hidden", activation=None))
                if not self.is_test:
                    self.losses = self.get_loss(output)
                    self.train_op = self.apply_gradients(opt, self.losses)
                    probs = tf.nn.softmax(output)
                    self.preds = tf.argmax(probs, 1)
            else:
                output = tf.squeeze(tf.layers.dense(hidden, units=1, name="output_hidden", activation=None))
                if not self.is_test:
                    self.losses = self.get_loss(output)
                    self.train_op = self.apply_gradients(opt, self.losses)
                self.preds = tf.nn.sigmoid(output)

    def get_loss(self, output):
        if self.loss_function == "softmax":
            l = tf.losses.sparse_softmax_cross_entropy(self.pred_labels, output)
        else:
            l = tf.losses.sigmoid_cross_entropy(self.pred_labels, output)
        l = tf.reduce_mean(l)
        # for v in tf.trainable_variables():
        #     if not 'bias' in v.name.lower():
        #         l += 0.0001 * tf.nn.l2_loss(v)
        return l
    
    def apply_gradients(self, opt, losses):
        gd = opt.compute_gradients(self.losses)
        return opt.apply_gradients(gd)
    
    def get_attention(self, inputs):
        # batch_size x length x 128
        cl_ = tf.layers.dense(inputs, units=self.hidden_layer_size, name="hidden_layer", activation=tf.nn.relu)
        # batch_size x self.claim_length
        cl_s_ = tf.squeeze(tf.layers.dense(cl_, units=1, name="hidden_softmax", activation=None))
        cl_s_ = tf.nn.softmax(cl_s_)
        cl_a = tf.multiply(tf.transpose(cl_, [2, 0, 1]), cl_s_)
        cl_a = tf.reduce_sum(tf.transpose(cl_a, [1, 2, 0]), axis=1)
        return cl_a

    def run_epochs(self, data, session, num_epoch=0, train_writer=None, train=True):
        if not train:
            train_op = tf.no_op()
        else:
            train_op = self.train_op
        
        dt_length = len(data[0])
        total_steps = dt_length // self.batch_size
        total_loss = 0.0
        preds = []
        probs = []
        policy, claim, customer, labels = data

        for step in xrange(total_steps):
            index = range(step * self.batch_size, (step + 1) * self.batch_size)
            pt = policy[index]
            ct = customer[index]
            clt = claim[index]
            feed = {
                self.policy: pt,
                self.claim: clt,
                self.customer: ct,
            }
            if self.is_test:
                pred = session.run([self.preds], feed_dict=feed)
            else:
                feed[self.pred_labels] = labels[index]
                loss, pred, _= session.run([self.losses, self.preds, train_op], feed_dict=feed)

                total_loss += loss
                sys.stdout.write('\r{} / {} loss = {}'.format(
                    step, total_steps, total_loss / (step + 1)))
                sys.stdout.flush()
            
            if self.loss_function == "softmax":
                preds += [int(x) for x in pred]
            else:
                preds += [1 if x >= 0.5 else 0 for x in pred]
                probs += [float(x) for x in pred]
        
        if not self.is_test:
            sys.stdout.write("\r")
            end = total_steps * self.batch_size
            labels = labels[:end]
            score = f1_score(labels, preds, average="binary")
        else: 
            score = 0
        
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
        
        return total_loss, score, preds, probs
                
