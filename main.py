import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import somoclu
import deep_som as ds
import time
import matplotlib.cm as cm
# %matplotlib inlinepip

import pprint, pickle

slice = 1000

training_set = None
labels = None

print("loading training set")
with open('spectrograms_4_training.pickle', 'rb') as f:
    training_set = pickle.load(f)

with open('file_list_4.pickle', 'rb') as f:
    labels = pickle.load(f)

print("preparing training set")
for i, l in enumerate(labels):
    labels[i] = l.parent.stem

training_set = training_set[:slice]
labels = labels[:slice]

#









labels_set = set(labels)

cmap = plt.cm.get_cmap('RdBu')

colors_list = [cmap(float(i) / len(labels_set)) for i in range(len(labels_set))]

label_to_color = dict(zip(labels_set, colors_list))

colors = []
for i, l in enumerate(labels):
    colors.append(label_to_color[l])

training_set_flatten = np.transpose(training_set.reshape(training_set.shape[0], 257, 150), (1, 0, 2)).reshape(257, training_set.shape[0] * 150).T

print("init dsom")
dsom = ds.DeepSom()
dsom.add_layer(100, 100, 257)
dsom.add_layer(100, 100, 30)
dsom.add_layer(100, 100, 20)

print("training")
dsom.train(training_set_flatten, verbose=1)

print("saving som")
with open('som.pickle', 'wb') as f:
    pickle.dump(dsom, f, pickle.HIGHEST_PROTOCOL)
print("som saved")

print("plotting result")
dsom._layers["som_2"].view_umatrix(colormap=cm.binary, bestmatches=True, bestmatchcolors=colors, labels=labels)
#
#
# num_data = 500
# num_test_data = 50
# c1 = np.random.rand(num_data, 3)/5
# c2 = (0.6, 0.1, 0.05) + np.random.rand(num_data, 3)/5
# c3 = (0.4, 0.1, 0.7) + np.random.rand(num_data, 3)/5
# data = np.float32(np.concatenate((c1, c2, c3)))
# test_data = np.float32(np.concatenate((c1[:num_test_data], c2[:num_test_data], c3[:num_test_data])))
# colors = ["red"] * num_data
# colors.extend(["green"] * num_data)
# colors.extend(["blue"] * num_data)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)
# labels = range(num_data*3)
# test_labels = range(num_test_data*3)
# test_colors = ["red"] * num_test_data
# test_colors.extend(["green"] * num_test_data)
# test_colors.extend(["blue"] * num_test_data)
#
# n_rows, n_columns = 100, 160
# som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False)
# time_start = time.time()
# som.train(data)
# time_end = time.time()
# print("elapsed time: ", time_end - time_start)
#
# am = som.get_surface_state(test_data)
# bmus = som.get_bmus(am)
# print(bmus.shape)

# som.update_data(test_data)
# som.train()
# print(som.bmus.shape)
# som.view_umatrix(colormap=cm.binary, bestmatches=True, bestmatchcolors=test_colors, labels=test_labels)

# som.view_activation_map(data_vector=test_data[0].reshape(1,3), colormap=cm.binary)
# plt.show()
# som.view_activation_map(data_vector=test_data[1].reshape(1,3))
# som.view_component_planes()
# plt.show()
