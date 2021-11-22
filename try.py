import json
from pathlib import Path

import numpy as np
from keras import datasets

import fed_learn

args = fed_learn.get_args()

fed_learn.set_working_GPU(str(args.gpu))

def model_fn():
    return fed_learn.create_model((32, 32, 3), 10, init_with_imagenet=False, learning_rate=args.learning_rate)

model = model_fn()
model.load_weights('global_weights4.h5')

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
data_handler = fed_learn.DataHandler(x_train, y_train, x_test, y_test, fed_learn.CifarProcessor(), args.debug)
#data_handler.assign_data_to_clients(server.clients, args.data_sampling_technique)
x_test, y_test = data_handler.preprocess(data_handler.x_test, data_handler.y_test)

results = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
print(results)