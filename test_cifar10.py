import nengo
import nengo_loihi
import nengo_dl

import gzip
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import keras
import scipy
import imageio
from stopwatch import Stopwatch
import cv2
import pickle_init
from pickle_init import PickleList

classes_list = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]

weights_file = "./lr001_b200_p4_e500_gray"
nr_epochs = 500
lr = 0.001
minibatch_size = 200
n_parallel = 4

out_size = 32 * 32 #* 3
inp_shape = (32, 32, 1)#, 3)
init_ones = (1, 1, 1, 1) # (1, 1, 3, 1)
test_data_size = 10000
nengo_loihi.set_defaults()

test_pickle_path = "CIFAR/augmented_cifar_test_gray.pkl"

'''
(train_images, train_labels), (test_images, test_labels) = (
    keras.datasets.cifar10.load_data())
'''
with open(test_pickle_path, 'rb') as f:
    test_data_obj = pickle.load(f, encoding="uint8")

test_data = test_data_obj.get_list()
#f = open(test_pickle_path, 'rb')
test_images = np.array(test_data[0])
test_labels = test_data[1]

# flatten images
test_images = test_images.reshape((test_images.shape[0], -1))

test_data = [test_images/254, test_labels]

#test_data[1] = [x[0] for x in test_data[1]]

one_hot = np.zeros((test_data_size, 10))
one_hot[np.arange(test_data_size), test_data[1]] = 1
test_data[1] = one_hot



def conv_layer(x, *args, activation=True, **kwargs):
    # create a Conv2D transform with the given arguments
    conv = nengo.Convolution(*args, channels_last=True, **kwargs)
    print(conv.output_shape.size)

    if activation:
        # add an ensemble to implement the activation function
        layer = nengo.Ensemble(conv.output_shape.size, 1).neurons
    else:
        # no nonlinearity, so we just use a node
        layer = nengo.Node(size_in=conv.output_shape.size)

    # connect up the input object to the new layer
    nengo.Connection(x, layer, transform=conv)

    # print out the shape information for our new layer
    print("LAYER")
    print(conv.input_shape.shape, "->", conv.output_shape.shape)

    return layer, conv


dt = 0.001  # simulation timestep
presentation_time = 0.1  # input presentation time
max_rate = 100  # neuron firing rates
# neuron spike amplitude (scaled so that the overall output is ~1)
amp = 1 / max_rate
# input image shape
input_shape = inp_shape

with nengo.Network(seed=0) as net:
    # set up the default parameters for ensembles/connections
    nengo_loihi.add_params(net)
    net.config[nengo.Ensemble].neuron_type = (
        nengo.SpikingRectifiedLinear(amplitude=amp))
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None

    # the input node that will be used to feed in input images
    inp = nengo.Node(
        nengo.processes.PresentInput(test_data[0], presentation_time),
        size_out=out_size)

    # the output node provides the 10-dimensional classification
    out = nengo.Node(size_in=10)
    for _ in range(n_parallel):
        # build parallel copies of the network
        layer, conv = conv_layer(
            inp, 1, input_shape, kernel_size=(1, 1),
            init=np.ones(init_ones))
        # first layer is off-chip to translate the images into spikes
        net.config[layer.ensemble].on_chip = False
        layer, conv = conv_layer(layer, 1, conv.output_shape,
                                 strides=(2, 2))
        nengo.Connection(layer, out, transform=nengo_dl.dists.Glorot())

    out_p = nengo.Probe(out)
    out_p_filt = nengo.Probe(out, synapse=nengo.Alpha(0.01))


#train_data = {inp: train_data[0][:, None, :],
#              out_p: train_data[1][:, None, :]}

# for the test data evaluation we'll be running the network over time
# using spiking neurons, so we need to repeat the input/target data
# for a number of timesteps (based on the presentation_time)
test_data = {
    inp: np.tile(test_data[0][:1000, None, :],
                 (1, int(presentation_time / dt), 1)),
    out_p_filt: np.tile(test_data[1][:1000, None, :],
                        (1, int(presentation_time / dt), 1))
}


def crossentropy(outputs, targets):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=outputs, labels=targets))


def classification_error(outputs, targets):
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                             tf.argmax(targets[:, -1], axis=-1)),
                tf.float32))


with nengo_dl.Simulator(net, minibatch_size=minibatch_size, seed=0) as sim:
    sim.load_params(weights_file)
    # store trained parameters back into the network
    sim.freeze_params(net)


for conn in net.all_connections:
    conn.synapse = 0.015


stopwatch_main = Stopwatch()
stopwatch_main.start()
n_presentations = 100
with nengo_loihi.Simulator(net, dt=dt, precompute=False) as sim:
    # if running on Loihi, increase the max input spikes per step
    if 'loihi' in sim.sims:
        print('on Loihi!')
        sim.sims['loihi'].snip_max_spikes_per_step = 360 # standard = ???

    # run the simulation on Loihi
    sim.run(n_presentations * presentation_time)
    stopwatch_main.stop()
    print("finished simulation in %s seconds" % stopwatch_main.duration)
    # check classification error
    step = int(presentation_time / dt)
    output = sim.data[out_p_filt][step - 1::step]
    correct = 100 * (np.mean(
        np.argmax(output, axis=-1)
        != np.argmax(test_data[out_p_filt][:n_presentations, -1],
                     axis=-1)
    ))
    print("loihi error: %.2f%%" % correct)



with nengo_dl.Simulator(net) as sim:
    # run the simulation on Loihi
    sim.run(n_presentations * presentation_time)
    stopwatch_main.stop()
    print("finished simulation in %s seconds" % stopwatch_main.duration)
    # check classification error
    step = int(presentation_time / dt)
    output = sim.data[out_p_filt][step - 1::step]
    correct = 100 * (np.mean(
        np.argmax(output, axis=-1)
        != np.argmax(test_data[out_p_filt][:n_presentations, -1],
                     axis=-1)
    ))
    print("nengo dl error: %.2f%%" % correct)
