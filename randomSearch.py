# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf
import random
import os
import copy

import reader
import util

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "random",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_float("init_scale", None,
                    "The initial scale of the weights.")
flags.DEFINE_bool("use_adam", False,
                  "Use Adam optimizer instead of SGD."
                  "If True, reduce the default learning rate to 1%.")
flags.DEFINE_float("learning_rate", None,
                    "The initial value of the learning rate.")
flags.DEFINE_float("max_grad_norm", None,
                    "The maximum permissible norm of the gradient.")
flags.DEFINE_integer("num_layers", None,
                    "The number of LSTM layers.")
flags.DEFINE_integer("num_steps", None,
                    "The number of unrolled steps of LSTM.")
flags.DEFINE_integer("hidden_size", None,
                    "The number of LSTM units.")
flags.DEFINE_integer("max_epoch", None,
                    "The number of epochs trained with the initial learning rate.")
flags.DEFINE_integer("max_max_epoch", None,
                    "The total number of epochs for training.")
flags.DEFINE_float("keep_prob", None,
                    "The probability of keeping weights in the dropout layer.")
flags.DEFINE_float("lr_decay", None,
                    'The decay of the learning rate for each epoch after "max_epoch".')
flags.DEFINE_integer("batch_size", None,
                    "The batch size.")
flags.DEFINE_integer("vocab_size", None,
                    "The size of vocabulary.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
flags.DEFINE_bool("use_dynamic", False,
                    "Use dynamic rnn instead of static rnn.")
flags.DEFINE_integer("n_experts", None,
                    "The number of softmax function in mixture of softmax."
                    "If equal to 0, use normal softmax function.")
flags.DEFINE_integer("num_options", 2,
                    "The number of config options in random search.")
flags.DEFINE_integer("num_tops", 1,
                    "The number of top config candidates in random search.")

FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    self._params_size = 0
    # self._memory_use = 0
    size = config.hidden_size
    vocab_size = config.vocab_size
    n_experts = config.n_experts

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    output, state = self._build_rnn_graph(inputs, config, is_training)

    if n_experts == 0:
      softmax_w = tf.get_variable(
          "softmax_w", [size, vocab_size], dtype=data_type())
      softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
      logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
      # Reshape logits to be a 3-D tensor for sequence loss
      logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
    else:
      # Mixture of Softmaxes
      prior_w = tf.get_variable("prior_w", [size, n_experts], dtype=data_type())
      prior_logit = tf.matmul(output, prior_w)
      prior = tf.nn.softmax(prior_logit)

      latent_w = tf.get_variable("latent_w", [size, n_experts*size], dtype=data_type())
      latent_b = tf.get_variable("latent_b", [n_experts*size], dtype=data_type())
      latent = tf.tanh(tf.nn.xw_plus_b(output, latent_w, latent_b))
      if is_training and config.keep_prob < 1:
        latent = tf.nn.dropout(latent, config.keep_prob)
      latent = tf.reshape(latent, [-1, size])

      logit_w = tf.get_variable("logit_w", [size, vocab_size], dtype=data_type())
      logit_b = tf.get_variable("logit_b", [vocab_size], dtype=data_type())
      logit = tf.nn.xw_plus_b(latent, logit_w, logit_b)
      prob = tf.nn.softmax(logit)
      prob = tf.reshape(prob, [-1, n_experts, vocab_size])
      prob = tf.reduce_sum(tf.transpose(prob, [2,0,1])*prior, axis=2)
      prob = tf.transpose(prob, [1,0])
      prob = tf.reshape(prob, [self.batch_size, self.num_steps, vocab_size])

      epsilon = tf.convert_to_tensor(1e-7, prob.dtype.base_dtype)
      prob = tf.clip_by_value(prob, epsilon, 1-epsilon)
      logits = tf.log(prob)

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True)

    # Update the cost
    self._cost = tf.reduce_sum(loss)
    self._final_state = state
    # self._memory_use = tf.contrib.memory_stats.MaxBytesInUse()

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)

    optimizer = tf.train.AdamOptimizer(self._lr) if config.use_adam else tf.train.GradientDescentOptimizer(self._lr)

    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)


  def _build_rnn_graph(self, inputs, config, is_training):
    if config.rnn_mode == CUDNN:
      return self._build_rnn_graph_cudnn(inputs, config, is_training)
    else:
      return self._build_rnn_graph_lstm(inputs, config, is_training)

  def _build_rnn_graph_cudnn(self, inputs, config, is_training):
    """Build the inference graph using CUDNN cell."""
    inputs = tf.transpose(inputs, [1, 0, 2])
    self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=config.num_layers,
        num_units=config.hidden_size,
        input_size=config.hidden_size,
        dropout=1 - config.keep_prob if is_training else 0)
    params_size_t = self._cell.params_size()
    self._rnn_params = tf.get_variable(
        "lstm_params",
        initializer=tf.random_uniform(
            [params_size_t], -config.init_scale, config.init_scale),
        validate_shape=False)
    c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
    outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, config.hidden_size])
    self._params_size = self._cell.params_size()
    return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

  def _get_lstm_cell(self, config, is_training):
    if config.rnn_mode == BASIC:
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)
    if config.rnn_mode == BLOCK:
      return tf.contrib.rnn.LSTMBlockCell(
          config.hidden_size, forget_bias=0.0)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell(config, is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(config.batch_size, data_type())
    state = self._initial_state
    # Simplified version of tf.nn.static_rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
    #
    # The alternative version of the code below is:
    #
    if config.use_dynamic:
        inputs = tf.transpose(inputs, [1, 0, 2])
        outputs, state = tf.nn.dynamic_rnn(cell, inputs,
                                            initial_state=self._initial_state,
                                            time_major=True)
    else:
        inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
        outputs, state = tf.nn.static_rnn(cell, inputs, initial_state=self._initial_state)
        # outputs = []
        # with tf.variable_scope("RNN"):
        #   for time_step in range(self.num_steps):
        #     if time_step > 0: tf.get_variable_scope().reuse_variables()
        #     (cell_output, state) = cell(inputs[:, time_step, :], state)
        #     outputs.append(cell_output)

    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])

    return output, state

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def export_ops(self, name):
    """Exports ops to collections."""
    self._name = name
    ops = {util.with_prefix(self._name, "cost"): self._cost}
    if self._is_training:
      ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update, params_size=self._params_size)
      if self._rnn_params:
        ops.update(rnn_params=self._rnn_params)
    for name, op in ops.items():
      tf.add_to_collection(name, op)
    self._initial_state_name = util.with_prefix(self._name, "initial")
    self._final_state_name = util.with_prefix(self._name, "final")
    util.export_state_tuples(self._initial_state, self._initial_state_name)
    util.export_state_tuples(self._final_state, self._final_state_name)

  def import_ops(self):
    """Imports ops from collections."""
    if self._is_training:
      self._train_op = tf.get_collection_ref("train_op")[0]
      self._lr = tf.get_collection_ref("lr")[0]
      self._new_lr = tf.get_collection_ref("new_lr")[0]
      self._lr_update = tf.get_collection_ref("lr_update")[0]
      self._params_size = tf.get_collection_ref("params_size")[0]
    #   self._memory_use = tf.get_collection_ref("memory_use")[0]
      rnn_params = tf.get_collection_ref("rnn_params")
    #   if self._cell and rnn_params:
        # params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
        #     self._cell,
        #     self._cell.params_to_canonical,
        #     self._cell.canonical_to_params,
        #     rnn_params,
        #     base_variable_scope="Model/RNN")
        # params_saveable = tf.contrib.cudnn_rnn.CudnnLSTMSaveable(
        #     rnn_params,
        #     self._cell.num_layers,
        #     self._cell.num_units,
        #     self._cell.input_size,
        #     self._cell.input_mode,
        #     self._cell.direction,
        #     scope="Model/RNN")
        # tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
    self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
    num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
    self._initial_state = util.import_state_tuples(
        self._initial_state, self._initial_state_name, num_replicas)
    self._final_state = util.import_state_tuples(
        self._final_state, self._final_state_name, num_replicas)

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name

  @property
  def params_size(self):
    return self._params_size

  # @property
  # def memory_use(self):
  #   return self._memory_use

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  use_adam = False
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK
  use_dynamic = False
  n_experts = 5


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  use_adam = False
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK
  use_dynamic = False
  n_experts = 10


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  use_adam = False
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK
  use_dynamic = False
  n_experts = 15


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  use_adam = False
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK
  use_dynamic = False
  n_experts = 2

class RandomConfig(object):
  """Random config, for tuning hyperparameters."""
  def __init__(self):
      self.init_scale = random.uniform(0., 0.1)
      self.use_adam = False
      self.learning_rate = 1.0
      self.max_grad_norm = random.uniform(5,10)
      self.num_layers = random.randint(1,3)
      self.max_epoch = 20
      self.max_max_epoch = 20
      self.keep_prob = random.uniform(0.3, 0.5)
      self.lr_decay = random.uniform(0.75,1.)
      self.vocab_size = 10000
      self.n_experts = random.randint(2,20)

      while True:
          self.num_steps = random.randrange(20,51,5)
          self.batch_size = random.randrange(10,31,2)
          self.hidden_size = random.randrange(500,1501,100)
          if self.num_steps*self.batch_size*self.hidden_size<=1200000:
              break

      self.rnn_mode_choice = random.randint(0,2)
      if self.rnn_mode_choice == 0:
          self.rnn_mode = BASIC
      elif self.rnn_mode_choice == 1:
          self.rnn_mode = BLOCK
      else:
          self.rnn_mode = CUDNN

      self.use_dynamic = False if random.randint(0,1) else True


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    # if verbose and step % (model.input.epoch_size // 10) == 10:
    #   print("%.3f perplexity: %.3f speed: %.0f wps" %
    #         (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
    #          iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
    #          (time.time() - start_time)), flush=True)

  return np.exp(costs / iters)


def get_config():
  """Get model config."""
  config = None
  if FLAGS.model == "small":
    config = SmallConfig()
  elif FLAGS.model == "medium":
    config = MediumConfimask
  elif FLAGS.model == "large":
    config = LargeConfig()
  elif FLAGS.model == "random":
    config = RandomConfig()
  elif FLAGS.model == "test":
    config = TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  if FLAGS.rnn_mode is not None:
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC
  if FLAGS.use_dynamic:
      config.use_dynamic = True
  if FLAGS.init_scale is not None:
      config.init_scale = FLAGS.init_scale
  if FLAGS.use_adam:
      config.use_adam = True
      config.learning_rate *= 0.01
  if FLAGS.learning_rate is not None:
      config.learning_rate = FLAGS.learning_rate
  if FLAGS.max_grad_norm is not None:
      config.max_grad_norm = FLAGS.max_grad_norm
  if FLAGS.num_layers is not None:
      config.num_layers = FLAGS.num_layers
  if FLAGS.num_steps is not None:
      config.num_steps = FLAGS.num_steps
  if FLAGS.hidden_size is not None:
      config.hidden_size = FLAGS.hidden_size
  if FLAGS.max_epoch is not None:
      config.max_epoch = FLAGS.max_epoch
  if FLAGS.max_max_epoch is not None:
      config.max_max_epoch = FLAGS.max_max_epoch
  if FLAGS.keep_prob is not None:
      config.keep_prob = FLAGS.keep_prob
  if FLAGS.lr_decay is not None:
      config.lr_decay = FLAGS.lr_decay
  if FLAGS.batch_size is not None:
      config.batch_size = FLAGS.batch_size
  if FLAGS.vocab_size is not None:
      config.vocab_size = FLAGS.vocab_size
  if FLAGS.n_experts is not None:
    config.n_experts = FLAGS.n_experts

  return config


def store_config(config):
    config_dic = {'init_scale': config.init_scale,
                'use_adam': config.use_adam,
                'learning_rate': config.learning_rate,
                'max_grad_norm': config.max_grad_norm,
                'num_layers': config.num_layers,
                'num_steps': config.num_steps,
                'hidden_size': config.hidden_size,
                'max_epoch': config.max_epoch,
                'max_max_epoch': config.max_max_epoch,
                'keep_prob': config.keep_prob,
                'lr_decay': config.lr_decay,
                'batch_size': config.batch_size,
                'vocab_size': config.vocab_size,
                'rnn_mode': config.rnn_mode,
                'use_dynamic': config.use_dynamic,
                'n_experts': config.n_experts,
                }

    return config_dic

def print_config(config):
    print('init_scale={}'.format(config.init_scale))
    print('use_adam={}'.format(config.use_adam))
    print('learning_rate={}'.format(config.learning_rate))
    print('max_grad_norm={}'.format(config.max_grad_norm))
    print('num_layers={}'.format(config.num_layers))
    print('num_steps={}'.format(config.num_steps))
    print('hidden_size={}'.format(config.hidden_size))
    print('max_epoch={}'.format(config.max_epoch))
    print('max_max_epoch={}'.format(config.max_max_epoch))
    print('keep_prob={}'.format(config.keep_prob))
    print('lr_decay={}'.format(config.lr_decay))
    print('batch_size={}'.format(config.batch_size))
    print('vocab_size={}'.format(config.vocab_size))
    print('rnn_mode={}'.format(config.rnn_mode))
    print('use_dynamic={}'.format(config.use_dynamic))
    print('n_experts={}'.format(config.n_experts))
    print('num_gpus={}'.format(FLAGS.num_gpus), flush=True)

def print_config_dir(config_dir):
    print('init_scale={}'.format(config_dir['init_scale']))
    print('use_adam={}'.format(config_dir['use_adam']))
    print('learning_rate={}'.format(config_dir['learning_rate']))
    print('max_grad_norm={}'.format(config_dir['max_grad_norm']))
    print('num_layers={}'.format(config_dir['num_layers']))
    print('num_steps={}'.format(config_dir['num_steps']))
    print('hidden_size={}'.format(config_dir['hidden_size']))
    print('max_epoch={}'.format(config_dir['max_epoch']))
    print('max_max_epoch={}'.format(config_dir['max_max_epoch']))
    print('keep_prob={}'.format(config_dir['keep_prob']))
    print('lr_decay={}'.format(config_dir['lr_decay']))
    print('batch_size={}'.format(config_dir['batch_size']))
    print('vocab_size={}'.format(config_dir['vocab_size']))
    print('rnn_mode={}'.format(config_dir['rnn_mode']))
    print('use_dynamic={}'.format(config_dir['use_dynamic']))
    print('n_experts={}'.format(config_dir['n_experts']))
    print('num_gpus={}'.format(FLAGS.num_gpus), flush=True)


def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


def train(configs, data):
  config, eval_config = configs
  train_data, valid_data, test_data = data

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)
      tf.summary.scalar("Test Loss", mtest.cost)

    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    for name, model in models.items():
      model.export_ops(name)
    metagraph = tf.train.export_meta_graph()
    if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
      raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                       "below 1.1.0")
    soft_placement = False
    if FLAGS.num_gpus > 1:
      soft_placement = True
      util.auto_parallel(metagraph, m)

  with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph)
    for model in models.values():
      model.import_ops()
    if FLAGS.save_path:
        if not os.path.isdir(FLAGS.save_path):
            os.mkdir(FLAGS.save_path)
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)

    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    best_pp = -1
    best_epoch = 0
    with sv.managed_session(config=config_proto) as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)
        train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
        valid_perplexity = run_epoch(session, mvalid)
        if best_pp == -1 or valid_perplexity < best_pp:
            best_pp = valid_perplexity
            best_epoch = i+1

  return best_pp, best_epoch


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  gpus = [
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  ]
  if FLAGS.num_gpus > len(gpus):
    raise ValueError(
        "Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))

  global_begin_time = time.time()

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data
  data = (train_data, valid_data, test_data)

  config_list = []
  val_pp_list = []
  ep_list = []

  for option in range(FLAGS.num_options):
    print("Rround "+str(option)+":")

    config = get_config()
    eval_config = copy.deepcopy(config)
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    configs = (config, eval_config)
    print_config(config)

    val_perplexity, ep = train(configs, data)
    print("Validation perplexity is "+str(val_perplexity)+'at epoch '+str(ep), flush=True)
    print("\n", flush=True)

    if len(val_pp_list) < FLAGS.num_tops or val_perplexity < min(val_pp_list):
        config_list.append(store_config(config))
        val_pp_list.append(val_perplexity)
        ep_list.append(ep)
        if len(val_pp_list) > FLAGS.num_tops:
            del config_list[val_pp_list.index(max(val_pp_list))]
            del ep_list[val_pp_list.index(max(val_pp_list))]
            del val_pp_list[val_pp_list.index(max(val_pp_list))]

  for config_dir, val_perplexity, ep in zip(config_list, val_pp_list, ep_list):
    print("One of config candidates is as follows:", flush=True)
    print_config_dir(config_dir)
    print("Validation perplexity is "+str(val_perplexity)+'at epoch '+str(ep), flush=True)
    print("\n")

if __name__ == "__main__":
  tf.app.run()
