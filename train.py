from __future__ import print_function

import os
import time
import argparse
import datetime
import logging
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from tqdm import tqdm

try:
    import cPickle
except ModuleNotFoundError:
    import pickle as cPickle

np.random.seed(72)
mx.random.seed(72)

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


class Identifier:
    def __init__(self):
        self.start = '{date:%Y_%m_%d_%H_%M_%S}'.format(date=datetime.datetime.now())

    def id(self, tag='xp'):
        return '{}_{}'.format(tag, self.start)


# noinspection PyProtectedMember
def get_mnist_data_from_gluon():
    train_data = gluon.data.vision.MNIST(train=True)
    train_label = train_data._label
    train_data = train_data._data.asnumpy().transpose(0, 3, 1, 2) / 255

    test_data = gluon.data.vision.MNIST(train=False)
    test_label = test_data._label
    test_data = test_data._data.asnumpy().transpose(0, 3, 1, 2) / 255

    train_data = np.array(train_data, dtype=np.float16)
    test_data = np.array(test_data, dtype=np.float16)

    train_mean = train_data.mean()
    train_data -= train_mean
    test_data -= train_mean
    return test_data, test_label, train_data, train_label


def get_cifar_data_from_gluon():
    train_data = gluon.data.vision.CIFAR10(train=True)
    train_label = train_data._label
    train_data = train_data._data.asnumpy().transpose(0, 3, 1, 2) / 255

    test_data = gluon.data.vision.CIFAR10(train=False)
    test_label = test_data._label
    test_data = test_data._data.asnumpy().transpose(0, 3, 1, 2) / 255

    train_mean = train_data.mean(axis=(0, 2, 3)).reshape(1, -1, 1, 1)
    train_data -= train_mean
    test_data -= train_mean
    return test_data, test_label, train_data, train_label


def get_training_data_batch(layer_data, ctx):
    data, label = layer_data.next()
    data = data.as_in_context(ctx)
    label = label.as_in_context(ctx)
    label_one_hot = nd.one_hot(label, num_classes)
    return data, label, label_one_hot


# Optimize 1 weight in a batch norm layer
def dim_1_random_switch(e, data, dim_1, label_one_hot, mask_name, net):
    start_idx = net.name_to_start[mask_name]
    initial_state = net.name_to_matrix[mask_name][dim_1].asscalar() * (1 - wd)
    net.name_to_matrix[mask_name][dim_1] = initial_state
    loss_switched_on = net.forward_and_return_loss(data, label_one_hot, start_idx)

    loss_list = [loss_switched_on]
    perturbation_list = [initial_state]

    perturbation = get_random_perturbation(net.layer_std[mask_name], e)
    net.name_to_matrix[mask_name][dim_1] = initial_state + perturbation
    loss_random = net.forward_and_return_loss(data, label_one_hot, start_idx)
    loss_list.append(loss_random)
    perturbation_list.append(perturbation + initial_state)
    net.name_to_matrix[mask_name][dim_1] = initial_state - perturbation
    loss_random = net.forward_and_return_loss(data, label_one_hot, start_idx)
    loss_list.append(loss_random)
    perturbation_list.append(initial_state - perturbation)

    loss_id = int(np.argmin(loss_list))
    flip = 0 if loss_id == 0 else 1
    net.name_to_matrix[mask_name][dim_1] = perturbation_list[loss_id]
    return flip, loss_list[0], loss_list[loss_id]


# Optimize 1 weight in a fully connected layer
def dim_2_random_switch(e, data, dim_1, dim_2, label_one_hot, mask_name, net):
    start_idx = net.name_to_start[mask_name]
    initial_state = net.name_to_matrix[mask_name][dim_1][dim_2].asscalar() * (1 - wd)
    net.name_to_matrix[mask_name][dim_1][dim_2] = initial_state
    loss_switched_on = net.forward_and_return_loss(data, label_one_hot, start_idx)

    loss_list = [loss_switched_on]
    perturbation_list = [initial_state]

    perturbation = get_random_perturbation(net.layer_std[mask_name], e)
    net.name_to_matrix[mask_name][dim_1][dim_2] = initial_state + perturbation
    loss_random = net.forward_and_return_loss(data, label_one_hot, start_idx)
    loss_list.append(loss_random)
    perturbation_list.append(perturbation + initial_state)
    net.name_to_matrix[mask_name][dim_1][dim_2] = initial_state - perturbation
    loss_random = net.forward_and_return_loss(data, label_one_hot, start_idx)
    loss_list.append(loss_random)
    perturbation_list.append(initial_state - perturbation)

    loss_id = int(np.argmin(loss_list))
    flip = 0 if loss_id == 0 else 1
    net.name_to_matrix[mask_name][dim_1][dim_2] = perturbation_list[loss_id]
    return flip, loss_list[0], loss_list[loss_id]


# Optimize 1 weight in a convolution layer
def dim_4_random_switch(e, data, dim_1, dim_2, dim_3, dim_4, label_one_hot, mask_name, net):
    start_idx = net.name_to_start[mask_name]
    initial_state = net.name_to_matrix[mask_name][dim_1][dim_2][dim_3][dim_4].asscalar() * (1 - wd)
    net.name_to_matrix[mask_name][dim_1][dim_2][dim_3][dim_4] = initial_state
    loss_switched_on = net.forward_and_return_loss(data, label_one_hot, start_idx)

    loss_list = [loss_switched_on]
    perturbation_list = [initial_state]

    perturbation = get_random_perturbation(net.layer_std[mask_name], e)
    net.name_to_matrix[mask_name][dim_1][dim_2][dim_3][dim_4] = initial_state + perturbation
    loss_random = net.forward_and_return_loss(data, label_one_hot, start_idx)
    loss_list.append(loss_random)
    perturbation_list.append(perturbation + initial_state)
    net.name_to_matrix[mask_name][dim_1][dim_2][dim_3][dim_4] = initial_state - perturbation
    loss_random = net.forward_and_return_loss(data, label_one_hot, start_idx)
    loss_list.append(loss_random)
    perturbation_list.append(initial_state - perturbation)

    loss_id = int(np.argmin(loss_list))
    flip = 0 if loss_id == 0 else 1
    net.name_to_matrix[mask_name][dim_1][dim_2][dim_3][dim_4] = perturbation_list[loss_id]
    return flip, loss_list[0], loss_list[loss_id]


def rso_train(net, ctx, data_set, sampling_batch_size, start=0, cid=0, output_first=True):
    # Data setup
    if data_set == "M10":
        test_data, test_label, train_data, train_label = get_mnist_data_from_gluon()
    else:
        test_data, test_label, train_data, train_label = get_cifar_data_from_gluon()

    # Data iterator setup
    cache_iter = ShuffleIter(train_data, train_label, 2000)
    test_iter = BasicIter(test_data, test_label, 2000)
    train_basic_iter = BasicIter(train_data, train_label, 2000)
    image_data = ShuffleIter(train_data, train_label, sampling_batch_size)
    layer_data = None

    best_test_acc, test_acc_list, epoch_start_time = -1, [], time.time()
    for e in range(start, total_epochs):
        logger.info("experiment id - {}".format(experiment_name.id()))

        # Optimize weights layer by layer
        for mask_name in net.lyr_optimization:
            mask_shape = net.name_to_matrix[mask_name].shape
            invert_filter_order = 0
            avg_loss, overall_flips = 0, 0

            if len(mask_shape) == 4:
                # Optimize weights in a convolution layers
                if mask_name == "conv_mask_0":
                    image_data.shuffle()
                    layer_data = image_data
                else:
                    cache_activations(cache_iter, net, ctx, train_basic_iter)
                    activation, label_act = net.get_data_loader(mask_name)
                    layer_data = ShuffleIter(activation, label_act, sampling_batch_size)

                output, input_channel = mask_shape[:2]
                if output_first:
                    outer, inner = output, input_channel
                else:
                    inner, outer = output, input_channel
                assert mask_shape[2] == mask_shape[3], "To use invert_filter_order"
                for idx_1 in tqdm(range(outer)):
                    for idx_2 in range(inner):
                        dim_1, dim_2 = (idx_1, idx_2) if output_first else (idx_2, idx_1)
                        for dim_3 in range(mask_shape[2]):
                            for dim_4 in range(mask_shape[3]):
                                dim_3, dim_4 = (dim_3, dim_4) if not invert_filter_order else (dim_4, dim_3)
                                data, _, label_one_hot = get_training_data_batch(layer_data, ctx)
                                flip, loss_switched_on, loss_random_val = dim_4_random_switch(e, data, dim_1, dim_2, dim_3, dim_4, label_one_hot, mask_name, net)
                                avg_loss += min(loss_switched_on, loss_random_val)
                                overall_flips += flip

            elif len(mask_shape) == 2:
                # Optimize weights in the fully connected (dense) layer
                cache_activations(cache_iter, net, ctx, train_basic_iter)
                activation, label_act = net.get_data_loader(mask_name)
                layer_data = ShuffleIter(activation, label_act, sampling_batch_size)

                input_channel, output = mask_shape[:2]  # Output is the 1th channel in FC layers, whereas it is the 0th channel on conv layers
                if output_first:
                    outer, inner = output, input_channel
                else:
                    inner, outer = output, input_channel
                for idx_2 in tqdm(range(outer)):
                    for idx_1 in range(inner):
                        dim_1, dim_2 = (idx_1, idx_2) if output_first else (idx_2, idx_1)
                        data, _, label_one_hot = get_training_data_batch(layer_data, ctx)
                        flip, loss_switched_on, loss_random_val = dim_2_random_switch(e, data, dim_1, dim_2, label_one_hot, mask_name, net)
                        avg_loss += min(loss_switched_on, loss_random_val)
                        overall_flips += flip

            else:
                # Optimize weights in the Batch Norm layers
                if "mv_" in mask_name:
                    # Skip weights for moving mean and moving variance
                    continue
                mask_shape = net.name_to_matrix[mask_name].shape
                layer_data.reset()

                for dim_1 in range(mask_shape[0]):
                    data, _, label_one_hot = get_training_data_batch(layer_data, ctx)
                    flip, loss_switched_on, loss_random_val = dim_1_random_switch(e, data, dim_1, label_one_hot, mask_name, net)
                    avg_loss += min(loss_switched_on, loss_random_val)
                    overall_flips += flip

            test_acc_list.append(evaluate_accuracy(test_iter, net, ctx, train_basic_iter))
            if test_acc_list[-1] > best_test_acc:
                best_test_acc = test_acc_list[-1]
            logger.info("e {}, mask {}, shape - {}, test_accuracy {}, best_test_acc {}".format(e, mask_name, mask_shape, test_acc_list[-1], best_test_acc))

        if cid == 0:
            e_time = time.time() - epoch_start_time
            required = (total_epochs - e - 1) * e_time
            logger.info("Time for last epoch - {}_hrs_{}_min".format(int(e_time//3600), int(e_time % 3600 // 60)))
            logger.info("Time left for experiment - {}_days_{}_hrs".format(int(required // 86400), int(required // 3600) % 24))
        epoch_start_time = time.time()
    return best_test_acc


# noinspection PyAttributeOutsideInit
# The CNN network class which runs on the mxnet ndarray imperative API
class CNN_ND:
    def __init__(self, ctx, input_filters, num_conv_layers, reverse_opt=True, sample_scale=1.0, opt_start=0, allow_partial_load=False):
        self.name_to_matrix = {}
        self.accepted_list = {}
        self.rejected_list = {}
        self.mask_dictionary = {}

        self.ctx = ctx
        self.num_classes = num_classes
        self.input_filters = input_filters
        self.reverse_optimization = reverse_opt
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = 1
        self.allow_partial_load = allow_partial_load
        self.opt_start = opt_start
        assert 0 <= self.opt_start <= (self.num_conv_layers + self.num_fc_layers - 1), "Optimization start must be within layer index bounds"

        self.grouped_conv = [0 for _ in range(self.num_conv_layers)]
        self.conv_size_per_layer = [3 for _ in range(self.num_conv_layers)]
        self.pad = [1 for _ in range(self.num_conv_layers)]
        self.previous_activation = [-1 for _ in range(self.num_conv_layers)]
        self.optimize_conv = [1 for _ in range(self.num_conv_layers)]
        self.pool_index = []
        self.kaiming_std = True
        self.sample_scale = sample_scale

        if self.num_conv_layers == 6:
            # The 6 layer network is the default network used for MNIST
            self.output_filters_per_layer = [16 for _ in range(self.num_conv_layers)]
            self.pool_index = [0, 1, 0, 1, 0, 0]

        elif self.num_conv_layers == 9:
            # The 9 layer network is the default network used for MNIST
            self.output_filters_per_layer = [16, 16, 16, 32, 32, 32, 32, 32, 32]
            self.pool_index = [0, 0, 1, 0, 1, 0, 0, 0, 0]
            self.previous_activation = [-1, -1, 0, -1, 2, -1, 4, -1, 6]

        else:
            # The number of layers in the network are used to map to architectures. Please add layer filters and information here.
            raise NotImplementedError("Please add layer filters and information here.")

        assert len(self.output_filters_per_layer) == len(self.conv_size_per_layer) == self.num_conv_layers == len(self.pad) == len(self.pool_index)
        logger.info("model description - {}\n".format(vars(self)))

        self.log_activation = False
        self.label_activation = []
        self.activation = [[] for _ in range(self.num_conv_layers)]

        self.initialize_params()

    def forward_and_return_loss(self, data, label_one_hot, start_idx=0, use_fp64=True):
        with autograd.record():
            output = self.forward(data, start_idx)
            loss = softmax_cross_entropy(output, label_one_hot, use_fp64)
        return nd.sum(loss).asscalar()

    def cpu_generate_then_move(self, deviation, shape):
        return nd.array(nd.random_normal(scale=deviation, ctx=mx.cpu(), shape=shape), ctx=self.ctx)

    def increase_filters_for_residual_connection(self, lyr_idx):
        return self.output_filters_per_layer[lyr_idx] != self.output_filters_per_layer[self.previous_activation[lyr_idx]]

    # Initialize the weights in the convolution layers using the He initialization and the sample scale
    def initialize_conv_layer(self, lyr_idx):
        self.name_to_matrix['gamma_{}'.format(lyr_idx)] = nd.ones(self.output_filters_per_layer[lyr_idx], ctx=self.ctx)
        self.name_to_matrix['mv_var_{}'.format(lyr_idx)] = nd.ones(self.output_filters_per_layer[lyr_idx], ctx=self.ctx)
        self.name_to_matrix['beta_{}'.format(lyr_idx)] = nd.zeros(self.output_filters_per_layer[lyr_idx], ctx=self.ctx)
        self.name_to_matrix['mv_mean_{}'.format(lyr_idx)] = nd.zeros(self.output_filters_per_layer[lyr_idx], ctx=self.ctx)

        input_channels = self.input_filters if lyr_idx == 0 else self.output_filters_per_layer[lyr_idx - 1]
        deviation = np.sqrt(2 / (input_channels * self.conv_size_per_layer[lyr_idx] * self.conv_size_per_layer[lyr_idx])) if self.kaiming_std else 0.75
        if self.grouped_conv[lyr_idx] != 0:
            assert input_channels % self.grouped_conv[lyr_idx] == 0
            input_channels = int(input_channels / self.grouped_conv[lyr_idx])
        wt_matrix = self.cpu_generate_then_move(deviation,
                                                (self.output_filters_per_layer[lyr_idx], input_channels, self.conv_size_per_layer[lyr_idx], self.conv_size_per_layer[lyr_idx]))
        self.name_to_matrix['conv_mask_{}'.format(lyr_idx)] = wt_matrix

        if self.previous_activation[lyr_idx] != -1 and self.increase_filters_for_residual_connection(lyr_idx):
            deviation = np.sqrt(2 / input_channels)
            wt_matrix = self.cpu_generate_then_move(deviation,
                                                    shape=(self.output_filters_per_layer[lyr_idx], self.output_filters_per_layer[self.previous_activation[lyr_idx]], 1, 1))
            self.name_to_matrix['filter_increase_{}'.format(lyr_idx)] = wt_matrix
            self.name_to_matrix['filter_increase_gamma_{}'.format(lyr_idx)] = nd.ones(self.output_filters_per_layer[lyr_idx], ctx=self.ctx)
            self.name_to_matrix['filter_increase_mv_var_{}'.format(lyr_idx)] = nd.ones(self.output_filters_per_layer[lyr_idx], ctx=self.ctx)
            self.name_to_matrix['filter_increase_beta_{}'.format(lyr_idx)] = nd.zeros(self.output_filters_per_layer[lyr_idx], ctx=self.ctx)
            self.name_to_matrix['filter_increase_mv_mean_{}'.format(lyr_idx)] = nd.zeros(self.output_filters_per_layer[lyr_idx], ctx=self.ctx)

    # Initialize the weights in the fully connected layer.
    def initialize_fc_layer(self):
        fc_feature_size = 1 * 1 * self.output_filters_per_layer[-1]
        fc_deviation = np.sqrt(2 / fc_feature_size) if self.kaiming_std else 2.0
        wt_matrix = self.cpu_generate_then_move(fc_deviation, shape=(fc_feature_size, self.num_classes))

        self.name_to_matrix['fc_mask_2'] = wt_matrix

    def initialize_params(self):
        mx.random.seed(420)
        for lyr_idx in range(self.num_conv_layers):
            self.initialize_conv_layer(lyr_idx)
        self.initialize_fc_layer()
        self.initialize_matrix_pointers()

    # Initialize meta data and pointers for the convolution and fully connected layers
    def initialize_matrix_pointers(self):
        self.layer_std = {}
        self.lyr_optimization = []
        self.name_to_start = {'fc_mask_2': self.num_conv_layers}

        for lyr_idx in range(self.num_conv_layers):
            input_channels = self.input_filters if lyr_idx == 0 else self.output_filters_per_layer[lyr_idx - 1]
            std_lyr = np.sqrt(2 / (input_channels * self.conv_size_per_layer[lyr_idx] * self.conv_size_per_layer[lyr_idx])) if self.kaiming_std else 2.0
            self.layer_std['conv_mask_{}'.format(lyr_idx)] = std_lyr * self.sample_scale

            if self.previous_activation[lyr_idx] == -1:
                self.name_to_start['conv_mask_{}'.format(lyr_idx)] = lyr_idx
            else:
                self.name_to_start['conv_mask_{}'.format(lyr_idx)] = self.previous_activation[lyr_idx] + 1
                if self.increase_filters_for_residual_connection(lyr_idx):
                    self.name_to_start['filter_increase_{}'.format(lyr_idx)] = self.previous_activation[lyr_idx] + 1
                    self.name_to_start['filter_increase_gamma_{}'.format(lyr_idx)] = self.previous_activation[lyr_idx] + 1
                    self.name_to_start['filter_increase_beta_{}'.format(lyr_idx)] = self.previous_activation[lyr_idx] + 1
                    self.layer_std['filter_increase_{}'.format(lyr_idx)] = std_lyr * self.sample_scale
                    self.layer_std['filter_increase_gamma_{}'.format(lyr_idx)] = 0.167 * self.sample_scale
                    self.layer_std['filter_increase_beta_{}'.format(lyr_idx)] = 0.167 * self.sample_scale
            self.name_to_start['gamma_{}'.format(lyr_idx)] = self.name_to_start['conv_mask_{}'.format(lyr_idx)]
            self.name_to_start['beta_{}'.format(lyr_idx)] = self.name_to_start['conv_mask_{}'.format(lyr_idx)]
            self.layer_std['gamma_{}'.format(lyr_idx)] = 0.167 * self.sample_scale
            self.layer_std['beta_{}'.format(lyr_idx)] = 0.167 * self.sample_scale

            if self.opt_start > lyr_idx:
                continue

            if self.reverse_optimization:
                if self.previous_activation[lyr_idx] != -1 and self.increase_filters_for_residual_connection(lyr_idx):
                    self.lyr_optimization.extend(['filter_increase_gamma_{}'.format(lyr_idx), 'filter_increase_beta_{}'.format(lyr_idx), 'filter_increase_{}'.format(lyr_idx)])
                self.lyr_optimization.extend(['gamma_{}'.format(lyr_idx), 'beta_{}'.format(lyr_idx)])
                if self.optimize_conv[lyr_idx]:
                    self.lyr_optimization.append('conv_mask_{}'.format(lyr_idx))
            else:
                if self.optimize_conv[lyr_idx]:
                    self.lyr_optimization.append('conv_mask_{}'.format(lyr_idx))
                self.lyr_optimization.extend(['beta_{}'.format(lyr_idx), 'gamma_{}'.format(lyr_idx)])
                if self.previous_activation[lyr_idx] != -1 and self.increase_filters_for_residual_connection(lyr_idx):
                    self.lyr_optimization.extend(['filter_increase_{}'.format(lyr_idx), 'filter_increase_beta_{}'.format(lyr_idx), 'filter_increase_gamma_{}'.format(lyr_idx)])

        fc_feature_size = 1 * 1 * self.output_filters_per_layer[-1]
        self.layer_std['fc_mask_2'] = np.sqrt(2 / fc_feature_size) if self.kaiming_std else 2.0
        self.layer_std['fc_mask_2'] = self.layer_std['fc_mask_2'] * self.sample_scale
        self.lyr_optimization.append('fc_mask_2')

        if self.reverse_optimization:
            self.lyr_optimization = self.lyr_optimization[::-1]

        self.layer_to_cache_idx = {'fc_mask_2': self.num_conv_layers - 1}
        for lyr_idx in range(1, self.num_conv_layers):
            self.layer_to_cache_idx['conv_mask_{}'.format(lyr_idx)] = self.name_to_start['conv_mask_{}'.format(lyr_idx)] - 1
            if self.previous_activation[lyr_idx] != -1 and self.increase_filters_for_residual_connection(lyr_idx):
                self.layer_to_cache_idx['filter_increase_{}'.format(lyr_idx)] = self.name_to_start['filter_increase_{}'.format(lyr_idx)] - 1

    # Forward pass. Supports residual connections and group conv.
    def forward(self, X, start_idx=0):
        convolution_mapping = {}
        for lyr_idx in range(start_idx, self.num_conv_layers):
            if lyr_idx > 0:
                convolution_mapping[lyr_idx - 1] = X
                X = nd.maximum(X, nd.zeros_like(X))  # relu

            if self.grouped_conv[lyr_idx] == 0:
                self.h1_conv = nd.Convolution(data=X, weight=self.name_to_matrix['conv_mask_{}'.format(lyr_idx)], bias=None, num_filter=self.output_filters_per_layer[lyr_idx],
                                              no_bias=True, kernel=(self.conv_size_per_layer[lyr_idx], self.conv_size_per_layer[lyr_idx]),
                                              pad=(self.pad[lyr_idx], self.pad[lyr_idx]))
            else:
                self.h1_conv = nd.Convolution(data=X, weight=self.name_to_matrix['conv_mask_{}'.format(lyr_idx)], bias=None, num_filter=self.output_filters_per_layer[lyr_idx],
                                              no_bias=True, kernel=(self.conv_size_per_layer[lyr_idx], self.conv_size_per_layer[lyr_idx]),
                                              pad=(self.pad[lyr_idx], self.pad[lyr_idx]), num_group=self.grouped_conv[lyr_idx])
            if self.previous_activation[lyr_idx] != -1:
                initial_activation = convolution_mapping[self.previous_activation[lyr_idx]]
                if self.increase_filters_for_residual_connection(lyr_idx):
                    initial_activation = nd.Convolution(data=initial_activation, weight=self.name_to_matrix['filter_increase_{}'.format(lyr_idx)],
                                                        bias=None, num_filter=self.output_filters_per_layer[lyr_idx],
                                                        no_bias=True, kernel=(1, 1), pad=(0, 0))
                    initial_activation = nd.BatchNorm(initial_activation, gamma=self.name_to_matrix['filter_increase_gamma_{}'.format(lyr_idx)],
                                                      beta=self.name_to_matrix['filter_increase_beta_{}'.format(lyr_idx)], momentum=0, fix_gamma=False, use_global_stats=False,
                                                      moving_mean=self.name_to_matrix['filter_increase_mv_mean_{}'.format(lyr_idx)],
                                                      moving_var=self.name_to_matrix['filter_increase_mv_var_{}'.format(lyr_idx)])
                self.h1_conv = self.h1_conv + initial_activation

            if self.pool_index[lyr_idx]:
                self.h1_conv = nd.Pooling(data=self.h1_conv, pool_type="avg", kernel=(2, 2), stride=(2, 2))
            self.h1_conv = nd.BatchNorm(self.h1_conv, gamma=self.name_to_matrix['gamma_{}'.format(lyr_idx)], beta=self.name_to_matrix['beta_{}'.format(lyr_idx)],
                                        momentum=0, fix_gamma=False, use_global_stats=False,
                                        moving_mean=self.name_to_matrix['mv_mean_{}'.format(lyr_idx)], moving_var=self.name_to_matrix['mv_var_{}'.format(lyr_idx)])

            X = self.h1_conv
            if self.log_activation:
                self.activation[lyr_idx].append(X.asnumpy())

        del convolution_mapping
        X = nd.Pooling(data=X, global_pool=True)
        self.c_out = nd.flatten(X)
        self.yhat_linear = nd.dot(self.c_out, self.name_to_matrix['fc_mask_2'])
        return self.yhat_linear

    def get_data_loader(self, mask_name):
        if mask_name not in self.layer_to_cache_idx:
            return None, None
        trdata = np.concatenate(self.activation[self.layer_to_cache_idx[mask_name]])
        return trdata, self.label_activation


# Sample a random perturbation for each weight from a gaussian with provided standard deviation
def get_random_perturbation(layer_std, e):
    std = layer_std * (ALPHA + (1 - ALPHA) * (total_epochs - e) / total_epochs)
    return np.random.normal(0.0, std)


# Very basic data iterator which loads all the data in the RAM
class BasicIter(mx.io.DataIter):
    def __init__(self, data, label, batch_size):
        super().__init__(batch_size)
        self.batch_size = batch_size
        self.data = data
        self.label = label
        self.data_size = len(self.data)

        assert self.data_size % self.batch_size == 0, 'There is no handling for roll over batches'
        self.total_index = int(np.floor(self.data_size / self.batch_size))
        self.batch_index = 0

    def reset(self):
        self.batch_index = 0

    def next(self):
        if self.batch_index == self.total_index:
            self.reset()
        start, end = self.batch_index * self.batch_size, self.batch_index * self.batch_size + self.batch_size
        data = nd.array(self.data[start:end])
        label = nd.array(self.label[start:end])
        self.batch_index += 1
        return data, label


# Shuffle enabled iterator
class ShuffleIter(mx.io.DataIter):
    def __init__(self, data, label, batch_size):
        super().__init__(batch_size)
        self.batch_size = batch_size
        self.data = data
        self.label = label
        self.num_images, _, _, _ = data.shape

        assert self.num_images % self.batch_size == 0
        self.total_index = int(np.floor(self.num_images / self.batch_size))
        self.batch_index = 0

    def reset(self):
        self.batch_index = 0

    def shuffle(self):
        mapping = np.random.permutation(self.num_images)
        self.data = self.data[mapping]
        self.label = self.label[mapping]

    def next(self):
        if self.batch_index == self.total_index:
            self.reset()
        start, end = self.batch_index * self.batch_size, self.batch_index * self.batch_size + self.batch_size
        data = nd.array(self.data[start:end])
        label = nd.array(self.label[start:end])
        self.batch_index += 1
        return data, label


def softmax_cross_entropy(yhat_linear, y, use_fp64=False):
    if use_fp64:
        yhat_linear = mx.nd.cast(yhat_linear, np.float64)
        y = mx.nd.cast(y, np.float64)
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)


def cache_activations(simple_iter, net, ctx, basic_iter):
    reset_bn_param(basic_iter, net, ctx)

    net.log_activation = True

    net.activation = [[] for _ in range(net.num_conv_layers)]
    net.label_activation = []

    data_list = []
    simple_iter.reset()
    simple_iter.shuffle()
    for index in range(simple_iter.total_index):
        data, label = simple_iter.next()
        data = data.as_in_context(ctx)
        net.forward(data)
        data_list.append(data.asnumpy())
        net.label_activation.append(label.asnumpy())

    net.label_activation = np.concatenate(net.label_activation)
    net.log_activation = False


# Reset the moving mean and the moving variance for all layers using the first batch on the training data set before evaluation.
def reset_bn_param(basic_iter, net, ctx):
    basic_iter.reset()
    data, label = basic_iter.next()
    data = data.as_in_context(ctx)

    # We use autograd.record for this forward pass otherwise the moving mean and the moving variance are not reset.
    with autograd.record():
        output = net.forward(data)
        # noinspection PyUnusedLocal
        predictions = nd.argmax(output, axis=1).asnumpy()


def evaluate_accuracy(simple_iter, net, ctx, train_cache=None):
    if train_cache:
        reset_bn_param(train_cache, net, ctx)
    numerator, denominator = 0., 0.
    simple_iter.reset()
    for index in range(simple_iter.total_index):
        data, label = simple_iter.next()
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        output = net.forward(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]

    return (numerator / denominator).asscalar()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
    parser.add_argument("--epochs", type=int, default=50, help='Total cycles to run')
    parser.add_argument("--layers", type=int, default=6, help='Number of convolution layers')
    parser.add_argument("--sample_batch", type=int, default=5000, help='Batch Size')
    parser.add_argument("--anneal", type=float, default=0.1, help='Should be less than 1. Linearly anneal sample deviation to given fraction.')
    parser.add_argument("--wd", type=float, default=0.0001, help='weight decay per step')
    parser.add_argument("--data", default='M10', choices=['M10', 'C10'], help="choose M10 for MNIST and C10 to CIFAR-10")
    args = parser.parse_args()

    # Constants setup
    num_classes = 10
    total_epochs = args.epochs
    ALPHA = args.anneal
    wd = args.wd
    experiment_name = Identifier()
    input_filters = 1 if args.data == "M10" else 3
    ctx = mx.gpu(args.gpu)

    # Logging setup
    logger = logging.getLogger("global")
    logger.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s::%(message)s')
    stream_formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler("logs/{}.log".format(experiment_name.id()))
    stream_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(args)

    # Initialize model
    net = CNN_ND(ctx, input_filters, args.layers)

    # Start RSO
    rso_train(net, ctx, args.data, args.sample_batch)
