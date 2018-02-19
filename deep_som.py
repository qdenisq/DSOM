import somoclu
import numpy as np


class DeepSom(object):
    """Class for training a sequence of self-organazing maps"""

    def __init__(self):
        """Constructor for the class.
                """
        self._n_layers = 0
        self._dim_layers = []
        self._layers = {}
        self._layers_input = {}
        return

    def add_layer(self, n_columns, n_rows, n_dim, initialcodebook=None,
                  kerneltype=0, maptype="planar", gridtype="rectangular",
                  compactsupport=True, neighborhood="gaussian", std_coeff=0.5,
                  initialization=None, data=None, verbose=0):
        som = somoclu.Somoclu(n_columns, n_rows, initialcodebook,
                              kerneltype, maptype, gridtype, compactsupport, neighborhood, std_coeff, initialization,
                              data, verbose)
        som.n_dim = n_dim
        som_name = "som_{}".format(self._n_layers)
        self._layers[som_name] = som
        self._dim_layers.append((n_rows, n_columns, n_dim))
        self._n_layers += 1

    def train(self, data=None, epochs=10, radius0=0, radiusN=1,
              radiuscooling="linear",
              scale0=0.1, scaleN=0.01, scalecooling="linear"):

        if data is None and "som_0" not in self._layers_input:
            raise Exception("The input data is missing")

        outputs = {}
        self._layers_input["som_0"] = data
        for i in range(self._n_layers):
            layer_name = "som_{}".format(i)
            som = self._layers[layer_name]
            # train
            som.train(self._layers_input[layer_name], epochs, radius0, radiusN, radiuscooling, scale0, scaleN,
                      scalecooling)
            # calculate activation map of input data after training
            am = som.get_surface_state(self._layers_input[layer_name])
            # find bmus
            bmus = som.get_bmus(am)
            outputs[layer_name] = bmus

            if i == self._n_layers - 1:
                return outputs

            # norm bmus according to the shape of the layer
            bmus = np.divide(bmus, [som._n_columns, som._n_rows])
            # flatten activation
            bmus = bmus.flatten()
            # reshape activation according to the input shape of the next layer
            next_layer_name = "som_{}".format(i + 1)
            n_input_dim = self._layers[next_layer_name].n_dim
            n_samples = len(bmus) % n_input_dim
            bmus = bmus.reshape(n_input_dim, n_samples)
            # assign activation to the next layer input data
            self._layers_input[next_layer_name] = bmus
        
