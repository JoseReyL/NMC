# %%


import gzip
import os
import pickle
from urllib.request import urlretrieve

import nengo
import nengo_dl
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    import requests

    has_requests = True
except ImportError:
    has_requests = False

import nengo_loihi
import keras

gpu_options = tf.GPUOptions(allow_growth=True)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def random_crop(img, crop_size=(10, 10)):
    assert crop_size[0] <= img.shape[0] and crop_size[1] <= img.shape[1], "Crop size should be less than image size"
    img = img.copy()
    w, h = img.shape[:2]
    x, y = np.random.randint(h - crop_size[0]), np.random.randint(w - crop_size[1])
    img = img[y:y + crop_size[0], x:x + crop_size[1]]
    return img


def center_crop(img, crop_size=(10, 10)):
    assert crop_size[0] <= img.shape[0] and crop_size[1] <= img.shape[1], "Crop size should be less than image size"
    img = img.copy()
    w, h = img.shape[:2]
    x, y = int((w - crop_size[0]) / 2), int((h - crop_size[1]) / 2)
    img = img[y:y + crop_size[0], x:x + crop_size[1]]
    return img


# %%

(train_images, train_labels), (test_images, test_labels) = (
    keras.datasets.cifar10.load_data())

# flatten
test_images = test_images.reshape((test_images.shape[0], -1))

test_data = [test_images / 254, test_labels]

# %%

test_data[1] = [x[0] for x in test_data[1]]

# %%

one_hot = np.zeros((test_data[0].shape[0], 10))
one_hot[np.arange(test_data[0].shape[0]), test_data[1]] = 1
test_data[1] = one_hot

# %%


temp = []
for img in test_data[0]:
    temp.append(center_crop(img.reshape(32, 32, 3), (24, 24)).reshape(-1))
test_data[0] = np.array(temp)
temp = []


# %%

def conv_layer(x, *args, activation=True, **kwargs):
    # create a Conv2D transform with the given arguments
    conv = nengo.Convolution(*args, channels_last=True, **kwargs)

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
    print(conv.output_shape.size)

    return layer, conv


# %%
dt = 0.001  # simulation timestep
presentation_time = 0.2  # input presentation time
max_rate = 200  # neuron firing rates
# neuron spike amplitude (scaled so that the overall output is ~1)
amp = 1 / max_rate
# input image shape
input_shape = (24, 24, 3)
n_parallel = 16
size_out = 20
nengo_loihi.set_defaults()

with nengo.Network(seed=0) as net:
    # set up the default parameters for ensembles/connections
    nengo_loihi.add_params(net)
    net.config[nengo.Ensemble].neuron_type = (
        nengo.LIF(amplitude=amp))
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None

    # the input node that will be used to feed in input images
    inp = nengo.Node(
        nengo.processes.PresentInput(test_data[0], presentation_time),
        size_out=24 * 24 * 3)

    # the output node provides the 10-dimensional classification
    out = nengo.Node(size_in=10)

    layer, conv = conv_layer(
        inp, 3, input_shape, kernel_size=(1, 1), strides=(1, 1),
        init=np.ones((1, 1, 3, 3)))

    # first layer is off-chip to translate the images into spikes
    net.config[layer.ensemble].on_chip = False

    # build parallel copies of the network
    for _ in range(n_parallel):
        layer2, conv2 = conv_layer(layer, 1, conv.output_shape, kernel_size=(3, 3),
                                   strides=(2, 2))

        nengo.Connection(layer2, out, transform=nengo_dl.dists.Glorot())

    print("LAYER")
    print(out.size_in, "->", out.size_out)
    out_p = nengo.Probe(out)
    out_p_filt = nengo.Probe(out, synapse=nengo.Alpha(0.016))



# set up training data
minibatch_size = 100


# for the test data evaluation we'll be running the network over time
# using spiking neurons, so we need to repeat the input/target data
# for a number of timesteps (based on the presentation_time)
test_data = {
    inp: np.tile(test_data[0][:100, None, :],
                 (1, int(presentation_time / dt), 1)),
    out_p_filt: np.tile(test_data[1][:100, None, :],
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



do_training = False

with nengo_dl.Simulator(net, minibatch_size=minibatch_size, seed=0) as sim:
    if do_training:
        # print("error before training: %.2f%%" %
        #      sim.loss(test_data, {out_p_filt: classification_error}))

        # run training
        sim.train(train_data, tf.train.RMSPropOptimizer(learning_rate=0.001),
                  objective={out_p: crossentropy}, n_epochs=5)

        sim.save_params("./cifar10")
    else:
        # download("mnist_params.data-00000-of-00001",
        #         "1BaNU7Er_Q3SJt4i4Eqbv1Ln_TkmmCXvy")
        # download("mnist_params.index", "1w8GNylkamI-3yHfSe_L1-dBtvaQYjNlC")
        # download("mnist_params.meta", "1JiaoxIqmRupT4reQ5BrstuILQeHNffrX")
        sim.load_params("./cifar10")

    # store trained parameters back into the network
    sim.freeze_params(net)



for conn in net.all_connections:
    conn.synapse = 0.016



n_presentations = 50
with nengo_loihi.Simulator(net, dt=dt, precompute=False) as sim:
    # if running on Loihi, increase the max input spikes per step
    if 'loihi' in sim.sims:
        sim.sims['loihi'].snip_max_spikes_per_step = 120000

    # run the simulation on Loihi
    sim.run(n_presentations * presentation_time)

    # check classification error
    step = int(presentation_time / dt)
    output = sim.data[out_p_filt][step - 1::step]
    correct = 100 * (np.mean(
        np.argmax(output, axis=-1)
        != np.argmax(test_data[out_p_filt][:n_presentations, -1],
                     axis=-1)
    ))
    print("loihi error: %.2f%%" % correct)

