import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import somoclu
import deep_som as ds
import time
import matplotlib.cm as cm
# %matplotlib inlinepip

import pprint, pickle
training_set = None
with open('spectrograms_training.pickle', 'rb') as f:
    training_set = pickle.load(f)

dsom = ds.DeepSom()
dsom.add_layer(100, 100, 1)
dsom.add_layer(100, 100, 15)
dsom.add_layer(100, 100, 10)
num_data = 500
num_test_data = 50
c1 = np.random.rand(num_data, 3)/5
c2 = (0.6, 0.1, 0.05) + np.random.rand(num_data, 3)/5
c3 = (0.4, 0.1, 0.7) + np.random.rand(num_data, 3)/5
data = np.float32(np.concatenate((c1, c2, c3)))
test_data = np.float32(np.concatenate((c1[:num_test_data], c2[:num_test_data], c3[:num_test_data])))
colors = ["red"] * num_data
colors.extend(["green"] * num_data)
colors.extend(["blue"] * num_data)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)
labels = range(num_data*3)
test_labels = range(num_test_data*3)
test_colors = ["red"] * num_test_data
test_colors.extend(["green"] * num_test_data)
test_colors.extend(["blue"] * num_test_data)

n_rows, n_columns = 100, 160
som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False)
time_start = time.time()
som.train(data)
time_end = time.time()
print("elapsed time: ", time_end - time_start)

am = som.get_surface_state(test_data)
bmus = som.get_bmus(am)
print(bmus.shape)

# som.update_data(test_data)
# som.train()
# print(som.bmus.shape)
# som.view_umatrix(colormap=cm.binary, bestmatches=True, bestmatchcolors=test_colors, labels=test_labels)

# som.view_activation_map(data_vector=test_data[0].reshape(1,3), colormap=cm.binary)
# plt.show()
# som.view_activation_map(data_vector=test_data[1].reshape(1,3))
# som.view_component_planes()
# plt.show()
