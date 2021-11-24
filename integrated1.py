from keras import backend as K
from tensorflow.keras import optimizers, losses, models, layers
from keras.applications.vgg16 import VGG16


def create_model(input_shape: tuple, nb_classes: int, init_with_imagenet: bool = False, learning_rate: float = 0.01):
  weights = None
  if init_with_imagenet:
      weights = "imagenet"

  model = VGG16(input_shape=input_shape,
                classes=nb_classes,
                weights=weights,
                include_top=False)
  # "Shallow" VGG for Cifar10
  x = model.get_layer('block3_pool').output
  x = layers.Flatten(name='Flatten')(x)
  x = layers.Dense(512, activation='relu')(x)
  x = layers.Dense(nb_classes)(x)
  x = layers.Softmax()(x)
  model = models.Model(model.input, x)

  loss = losses.categorical_crossentropy
  optimizer = optimizers.SGD(lr=learning_rate, decay=0.99)
  # optimizer = optimizers.Adam()

  model.compile('adam', loss, metrics=["accuracy"])
  return model


def set_model_weights(model: models.Model, weight_list):
  for i, symbolic_weights in enumerate(model.weights):
      weight_values = weight_list[i]
      K.set_value(symbolic_weights, weight_values)












from typing import Callable

from keras import models


class Client:
    def __init__(self, id: int):
        self.id = id
        self.model: models.Model = None
        self.x_train = None
        self.y_train = None

    def _init_model(self, model_fn: Callable, model_weights):
        model = model_fn()
        set_model_weights(model, model_weights)
        self.model = model

    def receive_data(self, x, y):
        self.x_train = x
        self.y_train = y

    def receive_and_init_model(self, model_fn: Callable, model_weights):
        self._init_model(model_fn, model_weights)

    def edge_train(self, client_train_dict: dict):
        if self.model is None:
            raise ValueError("Model is not created for client: {0}".format(self.id))

        hist = self.model.fit(self.x_train, self.y_train, **client_train_dict)
        return hist

    def reset_model(self):
        get_rid_of_the_models(self.model)








import random
from typing import List

import numpy as np
from keras import utils


def iid_data_indices(nb_clients: int, labels: np.ndarray):
    labels = labels.flatten()
    data_len = len(labels)
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    chunks = np.array_split(indices, nb_clients)
    return chunks


def non_iid_data_indices(nb_clients: int, labels: np.ndarray, nb_shards: int = 200):
    labels = labels.flatten()
    data_len = len(labels)

    indices = np.arange(data_len)
    indices = indices[labels.argsort()]

    shards = np.array_split(indices, nb_shards)
    random.shuffle(shards)
    shards_for_users = np.array_split(shards, nb_clients)
    indices_for_users = [np.hstack(x) for x in shards_for_users]

    return indices_for_users


class BaseDataProcessor:
    def __init__(self):
        pass

    @staticmethod
    def pre_process(x: np.ndarray, y: np.ndarray, nb_classes: int):
        raise NotImplementedError


class CifarProcessor(BaseDataProcessor):
    def __init__(self):
        super().__init__()

    @staticmethod
    def pre_process(x: np.ndarray, y: np.ndarray, nb_classes: int):
        y = utils.np_utils.to_categorical(y, nb_classes)
        x = x.astype(np.float32)
        x /= 255.0
        return x, y


class DataHandler:
    def __init__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_test: np.ndarray,
                 y_test: np.ndarray,
                 preprocessor: BaseDataProcessor,
                 only_debugging: bool = True):
        self._nb_classes = len(np.unique(y_train))
        self._preprocessor = preprocessor

        if only_debugging:
            x_train = x_train[:100]
            y_train = y_train[:100]
            x_test = x_test[:100]
            y_test = y_test[:100]

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def _sample(self, sampling_technique: str, nb_clients: int):
        if sampling_technique.lower() == "iid":
            sampler_fn = iid_data_indices
        else:
            sampler_fn = non_iid_data_indices
        client_data_indices = sampler_fn(nb_clients, self.y_train)
        return client_data_indices

    def preprocess(self, x, y):
        x, y = self._preprocessor.pre_process(x, y, self._nb_classes)
        return x, y

    def assign_data_to_clients(self, clients: List[Client], sampling_technique: str):
        sampled_data_indices = self._sample(sampling_technique, len(clients))
        for client, data_indices in zip(clients, sampled_data_indices):
            x = self.x_train[data_indices]
            y = self.y_train[data_indices]
            x, y = self.preprocess(x, y)
            client.receive_data(x, y)







from typing import List, Optional

import numpy as np


class WeightSummarizer:
    def __init__(self):
        pass

    def process(self,
                client_weight_list: List[List[np.ndarray]],
                global_weights: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        raise NotImplementedError()


class FedAvg(WeightSummarizer):
    def __init__(self, nu: float = 1.0):
        """
        Federated Averaging

        :param nu: Controls the summarized client join model fraction to the global model
        """

        super().__init__()
        self.nu = nu

    def process(self,
                client_weight_list: List[List[np.ndarray]],
                global_weights_per_layer: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        nb_clients = len(client_weight_list)
        weights_average = [np.zeros_like(w) for w in client_weight_list[0]]

        for layer_index in range(len(weights_average)):
            w = weights_average[layer_index]
            if global_weights_per_layer is not None:
                global_weight_mtx = global_weights_per_layer[layer_index]
            else:
                global_weight_mtx = np.zeros_like(w)
            for client_weight_index in range(nb_clients):
                client_weight_mtx = client_weight_list[client_weight_index][layer_index]

                # TODO: this step should be done at client side (client should send the difference of the weights)
                client_weight_diff_mtx = client_weight_mtx - global_weight_mtx

                w += client_weight_diff_mtx
            weights_average[layer_index] = (self.nu * w / nb_clients) + global_weight_mtx
        return weights_average


















from typing import Callable

import numpy as np
from keras import models
class Server:
    def __init__(self, model_fn: Callable,
                 weight_summarizer: WeightSummarizer,
                 nb_clients: int = 100,
                 client_fraction: float = 0.2):
        self.nb_clients = nb_clients
        self.client_fraction = client_fraction
        self.weight_summarizer = weight_summarizer

        # Initialize the global model's weights
        self.model_fn = model_fn
        model = self.model_fn()
        print("model mtric names ------------------------------- >?>>>>>          ",model.metrics_names)
        self.global_test_metrics_dict = {k: [] for k in model.metrics_names}
        self.global_test_metrics_dict['loss'] = []
        self.global_test_metrics_dict['accuracy'] = []
        self.global_model_weights = model.get_weights()
        get_rid_of_the_models(model)

        self.global_train_losses = []
        self.epoch_losses = []

        self.clients = []
        self.client_model_weights = []

        # Training parameters used by the clients
        self.client_train_params_dict = {"batch_size": 32,
                                         "epochs": 5,
                                         "verbose": 1,
                                         "shuffle": True}

    def _create_model_with_updated_weights(self) -> models.Model:
        model = self.model_fn()
        models.set_model_weights(model, self.global_model_weights)
        return model

    def send_model(self, client):
        client.receive_and_init_model(self.model_fn, self.global_model_weights)

    def init_for_new_epoch(self):
        # Reset the collected weights
        self.client_model_weights.clear()
        # Reset epoch losses
        self.epoch_losses.clear()

    def receive_results(self, client):
        client_weights = client.model.get_weights()
        self.client_model_weights.append(client_weights)
        client.reset_model()

    def create_clients(self):
        # Create all the clients
        for i in range(self.nb_clients):
            client = Client(i)
            self.clients.append(client)

    def summarize_weights(self):
        new_weights = self.weight_summarizer.process(self.client_model_weights)
        self.global_model_weights = new_weights

    def get_client_train_param_dict(self):
        return self.client_train_params_dict

    def update_client_train_params(self, param_dict: dict):
        self.client_train_params_dict.update(param_dict)

    def test_global_model(self, x_test: np.ndarray, y_test: np.ndarray):
        model = self._create_model_with_updated_weights()
        results = model.evaluate(x_test, y_test, batch_size=32, verbose=1)

        results_dict = dict(zip(model.metrics_names, results))
        for metric_name, value in results_dict.items():
            self.global_test_metrics_dict[metric_name].append(value)

        get_rid_of_the_models(model)

        return results_dict

    def select_clients(self):
        nb_clients_to_use = max(int(self.nb_clients * self.client_fraction), 1)
        client_indices = np.arange(self.nb_clients)
        np.random.shuffle(client_indices)
        selected_client_indices = client_indices[:nb_clients_to_use]
        return np.asarray(self.clients)[selected_client_indices]

    def save_model_weights(self, path: str):
        model = self._create_model_with_updated_weights()
        model.save_weights(str(path), overwrite=True)
        get_rid_of_the_models(model)

    def load_model_weights(self, path: str, by_name: bool = False):
        model = self._create_model_with_updated_weights()
        model.load_weights(str(path), by_name=by_name)
        self.global_model_weights = model.get_weights()
        get_rid_of_the_models(model)




























import gc
import os
from typing import List

# from keras import backend as K


def get_rid_of_the_models(model=None):
    """
    This function clears the TF session from the model.
    This is needed as TF/Keras models are not automatically cleared, and the memory will be overloaded
    """

    K.clear_session()
    if model is not None:
        del model
    gc.collect()


def print_selected_clients(clients: List[Client]):
    client_ids = [c.id for c in clients]
    print("Selected clients for epoch: {0}".format("| ".join(map(str, client_ids))))


def set_working_GPU(gpu_ids: str):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


































import json
from pathlib import Path

import numpy as np
from keras import datasets

# import fed_learn
import matplotlib.pyplot as plt
# args = fed_learn.get_args()
# args = {
#     "batch_size": 32,
#     "client_epochs": 1,
#     "clients": 100,
#     "data_sampling_technique": "iid",
#     "debug": False,
#     "fraction": 0.1,
#     "global_epochs": 1000,
#     "gpu": 0,
#     "learning_rate": 0.15,
#     "name": "iid",
#     "overwrite_experiment": False,
#     "weights_file": None
# }
def func(args):
  set_working_GPU(str(args['gpu']))

  experiment_folder_path = Path(__file__).resolve().parent
  experiment = fed_learn.Experiment(experiment_folder_path, True)
  experiment.serialize_args(args)

  #tf_scalar_logger = experiment.create_scalar_logger()

  client_train_params = {"epochs": args['client_epochs'], "batch_size": args['batch_size']}


  def model_fn():
      return create_model((32, 32, 3), 10, init_with_imagenet=False, learning_rate=args['learning_rate'])


  weight_summarizer = FedAvg()
  server = Server(model_fn,
                            weight_summarizer,
                            args['clients'],
                            args['fraction'])

  weight_path = args['weights_file']

  if weight_path is not None:
      server.load_model_weights(weight_path)

  server.update_client_train_params(client_train_params)
  server.create_clients()


  (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
  data_handler = DataHandler(x_train, y_train, x_test, y_test, CifarProcessor(), args['debug'])
  data_handler.assign_data_to_clients(server.clients, args['data_sampling_technique'])
  x_test, y_test = data_handler.preprocess(data_handler.x_test, data_handler.y_test)

  lossi = []
  acc = []

  plt.ion()
  # fig, ax = plt.subplots(2)
  for epoch in range(args['global_epochs']):
      print("Global Epoch {0} is starting".format(epoch))
      server.init_for_new_epoch()
      selected_clients = server.select_clients()
      print_selected_clients(selected_clients)

      for client in selected_clients:
          print("Client {0} is starting the training".format(client.id))
          server.send_model(client)
          hist = client.edge_train(server.get_client_train_param_dict())
          server.epoch_losses.append(hist.history["loss"][-1])
          server.receive_results(client)

      server.summarize_weights()

      epoch_mean_loss = np.mean(server.epoch_losses)
      server.global_train_losses.append(epoch_mean_loss)
      #tf_scalar_logger.log_scalar("train_loss/client_mean_loss", server.global_train_losses[-1], epoch)
      print("Loss (client mean): {0}".format(server.global_train_losses[-1]))

      global_test_results = server.test_global_model(x_test, y_test)
      print("--- Global test ---")
      test_loss = global_test_results["loss"]
      test_acc = global_test_results["accuracy"]
      print("{0}: {1}".format("Loss", test_loss))
      print("{0}: {1}".format("Accuracy", test_acc))
      lossi.append(test_loss)
      acc.append(test_acc)

      plt.plot(acc)
      # ax[0].title('model loss')
      # ax[0].ylabel('loss')
      # ax[0].xlabel('epoch')
      # ax[0].legend(['test'], loc='upper left')

      # ax[1].plot(acc)
      # ax[1].title('model loss')
      # ax[1].ylabel('accuracy')
      # ax[1].xlabel('epoch')
      # ax[1].legend(['test'], loc='upper right')
      plt.show()
      plt.pause(0.0001)
      plt.clf()
      #tf_scalar_logger.log_scalar("test_loss/global_loss", test_loss, epoch)
      #tf_scalar_logger.log_scalar("test_acc/global_acc", test_acc, epoch)

      with open(str(experiment.train_hist_path), 'w') as f:
          json.dump(server.global_test_metrics_dict, f)

      # TODO: save only when a condition is fulfilled (validation loss gets better, etc...)
      server.save_model_weights(experiment.global_weight_path)

      print("_" * 30)
