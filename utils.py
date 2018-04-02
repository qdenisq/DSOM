from __future__ import print_function

import numpy as np
import scipy.io.wavfile
from scipy.io.wavfile import read
from scipy import signal

import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.platform import gfile
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# boilerplate code
import os
from io import BytesIO
import numpy as np
import random
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import somoclu
import deep_som as ds
import time
import matplotlib.cm as cm
from pathlib import Path
import pprint, pickle


def extract_spectrogram(fname, nperseg=512, noverlap=384):
    sample_rate, samples = read(fname)
    frequencies, times, spectogram = signal.spectrogram(samples, sample_rate, nperseg=nperseg, noverlap=noverlap)
    dBS = 20 * np.log10(spectogram)
    dBS = scipy.stats.threshold(dBS, threshmin=0., newval=0.)
    return dBS


def init_cnn_model_settings():
    sample_rate = 16000
    clip_duration_ms = 1000
    window_size_ms = 30.
    window_stride_ms = 10.
    dct_coefficient_count = 40

    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length

    model_settings = {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        #      'fingerprint_size': fingerprint_size,
        #       'label_count': label_count,
        'sample_rate': sample_rate,
    }
    return model_settings


def load_speech_command_cnn(checkpoint_directory, checkpoint_name):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(os.path.join(checkpoint_directory, checkpoint_name + ".meta"))
    saver.restore(sess, os.path.join(checkpoint_directory, checkpoint_name))
    return sess


def restore_tensor(tensor_name):
    graph = tf.get_default_graph()
    op_to_restore = graph.get_tensor_by_name(tensor_name + ":0")
    return op_to_restore


def build_preproc_graph_for_cnn():
    model_settings = init_cnn_model_settings()
    wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
    audio_binary = tf.read_file(wav_data_placeholder)
    decoded_sample_data = contrib_audio.decode_wav(
        audio_binary,
        desired_channels=1,
        desired_samples=model_settings['desired_samples'],
        name='decoded_sample_datal')
    spectrogram = contrib_audio.audio_spectrogram(
        decoded_sample_data.audio,
        window_size=model_settings['window_size_samples'],
        stride=model_settings['window_stride_samples'],
        magnitude_squared=True)
    mfcc_tensor = contrib_audio.mfcc(
        spectrogram,
        decoded_sample_data.sample_rate,
        dct_coefficient_count=model_settings['dct_coefficient_count'])
    fingerprint_frequency_size = model_settings['dct_coefficient_count']
    fingerprint_time_size = model_settings['spectrogram_length']
    mfcc_tensor_flatten = tf.reshape(mfcc_tensor, [
        -1, fingerprint_time_size * fingerprint_frequency_size
    ])
    return mfcc_tensor_flatten, mfcc_tensor


def run_sess(sess ):

    return


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>" % size)
    return strip_def


def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add()
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0] != '^' else '^' + rename_func(s[1:])
    return res_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

