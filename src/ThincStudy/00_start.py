# %% markdown
# # Intro to Thinc for beginners: defining a simple model and config & wrapping PyTorch, TensorFlow and MXNet
#
# This example shows how to get started with Thinc, using the "hello world" of neural network models: recognizing handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). For comparison, here's the same model implemented in other frameworks: [PyTorch version](https://github.com/pytorch/examples/blob/master/mnist/main.py), [TensorFlow version](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py). In this notebook, we'll walk through **creating and training the model**, using **config files**, registering **custom functions** and **wrapping models** defined in PyTorch, TensorFlow and MXNet. This tutorial is aimed at beginners, but it assumes basic knowledge of machine learning concepts and terminology.
# %% codecell
# !pip install "thinc>=8.0.0a0" ml_datasets "tqdm>=4.41"
# %% markdown
# First, let's use Thinc's `prefer_gpu` helper to make sure we're performing operations **on GPU if available**. The function should be called right after importing Thinc, and it returns a boolean indicating whether the GPU has been activated.
# %% codecell
from thinc.api import prefer_gpu
prefer_gpu()
# %% markdown
# We’ve prepared a separate package [`ml-datasets`](https://github.com/explosion/ml-datasets) with loaders for some common datasets, including MNIST. So we can set up the data as follows:
# %% codecell
import ml_datasets
(train_X, train_Y), (dev_X, dev_Y) = ml_datasets.mnist()
print(f"Training size={len(train_X)}, dev size={len(dev_X)}")
# %% markdown
# Now let’s define a model with two **Relu-activated hidden layers**, followed by a **softmax-activated output layer**. We’ll also add **dropout** after the two hidden layers, to help the model generalize better. The `chain` combinator is like `Sequential` in PyTorch or Keras: it combines a list of layers together with a feed-forward relationship.
# %% codecell
from thinc.api import chain, Relu, Softmax

n_hidden = 32
dropout = 0.2

model = chain(
    Relu(nO=n_hidden, dropout=dropout),
    Relu(nO=n_hidden, dropout=dropout),
    Softmax()
)

model
# %% markdown
# After creating the model, we can call the `Model.initialize` method, passing in a small batch of input data `X` and a small batch of output data `Y`. This allows Thinc to **infer the missing dimensions**: when we defined the model, we didn’t tell it the input size `nI` or the output size `nO`. When passing in the data, make sure it is on the right device by calling `model.ops.asarray` which will e.g. transform the arrays to `cupy` when running on GPU.
# %% codecell
# making sure the data is on the right device
train_X = model.ops.asarray(train_X)
train_Y = model.ops.asarray(train_Y)
dev_X = model.ops.asarray(dev_X)
dev_Y = model.ops.asarray(dev_Y)

model.initialize(X=train_X[:5], Y=train_Y[:5])
nI = model.get_dim("nI")
nO = model.get_dim("nO")
print(f"Initialized model with input dimension nI={nI} and output dimension nO={nO}")
# %% markdown
# Next we need to create an **optimizer**, and make several passes over the data, randomly selecting paired batches of the inputs and labels each time. While some machine learning libraries provide a single `.fit()` method to train a model all at once, Thinc puts you in charge of **shuffling and batching your data**, with the help of a few handy utility methods. `model.ops.xp` is an instance of either `numpy` or `cupy`, depending on whether you run the code on CPU or GPU.
# %% codecell
from thinc.api import Adam, fix_random_seed
from tqdm.notebook import tqdm

fix_random_seed(0)
optimizer = Adam(0.001)
batch_size = 128
print("Measuring performance across iterations:")

for i in range(10):
    batches = model.ops.multibatch(batch_size, train_X, train_Y, shuffle=True)
    for X, Y in tqdm(batches, leave=False):
        Yh, backprop = model.begin_update(X)
        backprop(Yh - Y)
        model.finish_update(optimizer)
    # Evaluate and print progress
    correct = 0
    total = 0
    for X, Y in model.ops.multibatch(batch_size, dev_X, dev_Y):
        Yh = model.predict(X)
        correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
        total += Yh.shape[0]
    score = correct / total
    print(f" {i} {float(score):.3f}")
# %% markdown
# Let's wrap the training code in a function, so we can reuse it later:
# %% codecell
def train_model(data, model, optimizer, n_iter, batch_size):
    (train_X, train_Y), (dev_X, dev_Y) = data
    indices = model.ops.xp.arange(train_X.shape[0], dtype="i")
    for i in range(n_iter):
        batches = model.ops.multibatch(batch_size, train_X, train_Y, shuffle=True)
        for X, Y in tqdm(batches, leave=False):
            Yh, backprop = model.begin_update(X)
            backprop(Yh - Y)
            model.finish_update(optimizer)
        # Evaluate and print progress
        correct = 0
        total = 0
        for X, Y in model.ops.multibatch(batch_size, dev_X, dev_Y):
            Yh = model.predict(X)
            correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
            total += Yh.shape[0]
        score = correct / total
        print(f" {i} {float(score):.3f}")
# %% markdown
# ### Operator overloading for more concise model definitions
#
# Thinc allows you to **overload operators** and bind arbitrary functions to Python operators like `+`, `*`, but also `>>` or `@`. The `Model.define_operators` contextmanager takes a dict of operators mapped to functions – typically combinators like `chain`. The operators are only valid for the `with` block. This lets us define the model like this:
# %% codecell
from thinc.api import Model, chain, Relu, Softmax

n_hidden = 32
dropout = 0.2

with Model.define_operators({">>": chain}):
    model = Relu(nO=n_hidden, dropout=dropout) >> Relu(nO=n_hidden, dropout=dropout) >> Softmax()
# %% markdown
# If your model definitions are very complex, mapping combinators to operators can help you keep the code readable and concise. You can find more examples of model definitions with overloaded operators [in the docs](https://thinc.ai/docs). (Also note that you don't _have to_ use this syntax!)
# %% markdown
# ---
#
# ## Using config files
#
# Configuration is a huge problem for machine learning code, because you may want to expose almost any detail of any function as a hyperparameter. The setting you want to expose might be arbitrarily far down in your call stack. Default values also become hard to change without breaking backwards compatibility.
#
# To solve this problem, Thinc provides a config system that lets you easily describe **arbitrary trees of objects**. The objects can be created via function calls you register using a simple decorator syntax. The config can include values like hyperparameters or training settings (whatever you need), or references to functions and the values of their arguments. Thinc will then construct the config **bottom-up** – so you can define one function with its arguments, and then pass the return value into another function.
#
# > 💡 You can keep the config as a string in your Python script, or save it to a file like `config.cfg`. To load a config from a string, you can use `Config.from_str`. To load from a file, you can use `Config.from_disk`. The following examples all use strings so we can include them in the notebook.
# %% codecell
from thinc.api import Config, registry

EXAMPLE_CONFIG1 = """
[hyper_params]
learn_rate = 0.001

[optimizer]
@optimizers = "Adam.v1"
learn_rate = ${hyper_params:learn_rate}
"""

config1 = Config().from_str(EXAMPLE_CONFIG1)
config1
# %% markdown
# When you open the config with `Config.from_str`, Thinc will parse it as a dict and fill in the references to values defined in other sections. For example, `${hyper_params:learn_rate}` is substituted with `0.001`.
#
# Keys starting with `@` are references to **registered functions**. For example, `@optimizers = "Adam.v1"` refers to the function registered under the name `"Adam.v1"`, a function creating an Adam optimizer. The function takes one argument, the `learn_rate`. Calling `registry.make_from_config` will resolve the config and create the functions it defines.
# %% codecell
loaded_config1 = registry.make_from_config(config1)
loaded_config1
# %% markdown
# If function arguments are missing or have incompatible types, Thinc will raise an error and tell you what's wrong. Configs can also define **nested blocks** using the `.` notation. In this example, `optimizer.learn_rate` defines the `learn_rate` argument of the `optimizer` block. Instead of a float, the learning rate can also be a generator – for instance, a linear warm-up rate:
# %% codecell
EXAMPLE_CONFIG2 = """
[optimizer]
@optimizers = "Adam.v1"

[optimizer.learn_rate]
@schedules = "warmup_linear.v1"
initial_rate = 2e-5
warmup_steps = 1000
total_steps = 10000
"""

config2 = Config().from_str(EXAMPLE_CONFIG2)
config2
# %% markdown
# Calling `registry.make_from_config` will now construct the objects bottom-up: first, it will create the schedule with the given arguments. Next, it will create the optimizer and pass in the schedule as the `learn_rate` argument.
# %% codecell
loaded_config2 = registry.make_from_config(config2)
loaded_config2
# %% markdown
# This gives you a loaded optimizer using the settings defined in the config, which you can then use in your script. How you set up your config and what you do with the result is **entirely up to you**. Thinc just gives you a dictionary of objects back and makes no assumptions about what they _"mean"_. This means that you can also choose the names of the config sections – the only thing that needs to stay consistent are the names of the function arguments.
# %% markdown
# ### Configuring the MNIST model
#
# Here's a config describing the model we defined above. The values in the `hyper_params` section can be referenced in other sections to keep them consistent. The `*` is used for **positional arguments** – in this case, the arguments to the `chain` function, two Relu layers and one softmax layer.
# %% codecell
CONFIG = """
[hyper_params]
n_hidden = 32
dropout = 0.2
learn_rate = 0.001

[model]
@layers = "chain.v1"

[model.*.relu1]
@layers = "Relu.v1"
nO = ${hyper_params:n_hidden}
dropout = ${hyper_params:dropout}

[model.*.relu2]
@layers = "Relu.v1"
nO = ${hyper_params:n_hidden}
dropout = ${hyper_params:dropout}

[model.*.softmax]
@layers = "Softmax.v1"

[optimizer]
@optimizers = "Adam.v1"
learn_rate = ${hyper_params:learn_rate}

[training]
n_iter = 10
batch_size = 128
"""

config = Config().from_str(CONFIG)
config
# %% codecell
loaded_config = registry.make_from_config(config)
loaded_config
# %% markdown
# When you call `registry.make_from_config`, Thinc will first create the three layers using the specified arguments populated by the hyperparameters. It will then pass the return values (the layer objects) to `chain`. It will also create an optimizer. All other values, like the training config, will be passed through as a regular dict. Your training code can now look like this:
# %% codecell
model = loaded_config["model"]
optimizer = loaded_config["optimizer"]
n_iter = loaded_config["training"]["n_iter"]
batch_size = loaded_config["training"]["batch_size"]

model.initialize(X=train_X[:5], Y=train_Y[:5])
train_model(((train_X, train_Y), (dev_X, dev_Y)), model, optimizer, n_iter, batch_size)
# %% markdown
# If you want to change a hyperparamter or experiment with a different optimizer, all you need to change is the config. For each experiment you run, you can save a config and you'll be able to reproduce it later.
# %% markdown
# ---
#
# ## Programming via config vs. registering custom functions
#
# The config system is very powerful and lets you define complex relationships, including model definitions with levels of nested layers. However, it's not always a good idea to program entirely in your config – this just replaces one problem (messy and hard to maintain code) with another one (messy and hard to maintain configs). So ultimately, it's about finding the **best possible trade-off**.
#
# If you've written a layer or model definition you're happy with, you can use Thinc's function registry to register it and assign it a string name. Your function can take any arguments that can later be defined in the config. Adding **type hints** ensures that config settings will be **parsed and validated** before they're passed into the function, so you don't end up with incompatible settings and confusing failures later on. Here's the MNIST model, defined as a custom layer:
# %% codecell
import thinc

@thinc.registry.layers("MNIST.v1")
def create_mnist(nO: int, dropout: float):
    return chain(
        Relu(nO, dropout=dropout),
        Relu(nO, dropout=dropout),
        Softmax()
    )
# %% markdown
# In the config, we can now refer to it by name and set its arguments. This makes the config maintainable and compact, while still allowing you to change and record the hyperparameters.
# %% codecell
CONFIG2 = """
[model]
@layers = "MNIST.v1"
nO = 32
dropout = 0.2

[optimizer]
@optimizers = "Adam.v1"
learn_rate = 0.001

[training]
n_iter = 10
batch_size = 128
"""

config = Config().from_str(CONFIG2)
config
# %% codecell
loaded_config = registry.make_from_config(config)
loaded_config
# %% markdown
# If you don't want to hard-code the dataset being used, you can also wrap it in a registry function. This lets you refer to it by name in the config, and makes it easy to swap it out. In your config, you can then load the data in its own section, or as a subsection of `training`.
# %% codecell
@thinc.registry.datasets("mnist_data.v1")
def mnist():
    return ml_datasets.mnist()
# %% codecell
CONFIG3 = """
[model]
@layers = "MNIST.v1"
nO = 32
dropout = 0.2

[optimizer]
@optimizers = "Adam.v1"
learn_rate = 0.001

[training]
n_iter = 10
batch_size = 128

[training.data]
@datasets = "mnist_data.v1"
"""

config = Config().from_str(CONFIG3)
loaded_config = registry.make_from_config(config)
loaded_config
# %% codecell
model = loaded_config["model"]
optimizer = loaded_config["optimizer"]
n_iter = loaded_config["training"]["n_iter"]
batch_size = loaded_config["training"]["batch_size"]
(train_X, train_Y), (dev_X, dev_Y) = loaded_config["training"]["data"]

model.initialize(X=train_X[:5], Y=train_Y[:5])
train_model(((train_X, train_Y), (dev_X, dev_Y)), model, optimizer, n_iter, batch_size)
# %% markdown
# ---
#
# ## Wrapping TensorFlow, PyTorch and MXNet models
#
# The previous example showed how to define the model directly in Thinc, which is pretty straightforward. But you can also define your model using a **machine learning library of your choice** and wrap it as a Thinc model. This gives your layers a unified interface so you can easily mix and match them, and also lets you take advantage of the config system and type hints. Thinc currently ships with built-in wrappers for [PyTorch](https://pytorch.org), [TensorFlow](https://tensorflow.org) and [MXNet](https://mxnet.apache.org/).
# %% markdown
# ### Wrapping TensorFlow models
#
# Here's the same model definition in TensorFlow: a `Sequential` layer (equivalent of Thinc's `chain`) with two Relu layers and dropout, and an output layer with a softmax activation. Thinc's `TensorFlowWrapper` wraps the model and turns it into a regular Thinc `Model`.
# %% codecell
!pip install "tensorflow>2.0"
# %% codecell
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from thinc.api import TensorFlowWrapper, Adam

width = 32
nO = 10
nI = 784
dropout = 0.2

tf_model = Sequential()
tf_model.add(Dense(width, activation="relu", input_shape=(nI,)))
tf_model.add(Dropout(dropout))
tf_model.add(Dense(width, activation="relu", input_shape=(nI,)))
tf_model.add(Dropout(dropout))
tf_model.add(Dense(nO, activation="softmax"))

wrapped_tf_model = TensorFlowWrapper(tf_model)
wrapped_tf_model
# %% markdown
# You can now use the same training code to train the model:
# %% codecell
data = ml_datasets.mnist()
optimizer = Adam(0.001)
wrapped_tf_model.initialize(X=train_X[:5], Y=train_Y[:5])
train_model(data, wrapped_tf_model, optimizer, n_iter=10, batch_size=128)
# %% markdown
# ### Wrapping PyTorch models
#
# Here's the PyTorch version. Thinc's `PyTorchWrapper` wraps the model and turns it into a regular Thinc `Model`.
# %% codecell
!pip install torch
# %% codecell
import torch
import torch.nn
import torch.nn.functional as F
from thinc.api import PyTorchWrapper, Adam


width = 32
nO = 10
nI = 784
dropout = 0.2


class PyTorchModel(torch.nn.Module):
    def __init__(self, width, nO, nI, dropout):
        super(PyTorchModel, self).__init__()
        self.dropout1 = torch.nn.Dropout2d(dropout)
        self.dropout2 = torch.nn.Dropout2d(dropout)
        self.fc1 = torch.nn.Linear(nI, width)
        self.fc2 = torch.nn.Linear(width, nO)

    def forward(self, x):
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

wrapped_pt_model = PyTorchWrapper(PyTorchModel(width, nO, nI, dropout))
wrapped_pt_model
# %% markdown
# You can now use the same training code to train the model:
# %% codecell
data = ml_datasets.mnist()
optimizer = Adam(0.001)
wrapped_pt_model.initialize(X=train_X[:5], Y=train_Y[:5])
train_model(data, wrapped_pt_model, optimizer, n_iter=10, batch_size=128)
# %% markdown
# ### Wrapping MXNet models
#
# Here's the MXNet version. Thinc's `MXNetWrapper` wraps the model and turns it into a regular Thinc `Model`.
#
# MXNet doesn't provide a `Softmax` layer but a `.softmax()` operation/method for prediction and it integrates an internal softmax during training. So to be able to integrate it with the rest of the components, you combine it with a `Softmax()` Thinc layer using the `chain` combinator. Make sure you `initialize()` the MXNet model *and* the Thinc model.
# %% codecell
!pip install "mxnet>=1.5.1,<1.6.0"
# %% codecell
from mxnet.gluon.nn import Dense, Sequential, Dropout
from thinc.api import MXNetWrapper, chain, Softmax

width = 32
nO = 10
nI = 784
dropout = 0.2

mx_model = Sequential()
mx_model.add(Dense(width, activation="relu"))
mx_model.add(Dropout(dropout))
mx_model.add(Dense(width, activation="relu"))
mx_model.add(Dropout(dropout))
mx_model.add(Dense(nO))
mx_model.initialize()
wrapped_mx_model = chain(MXNetWrapper(mx_model), Softmax())
wrapped_mx_model
# %% markdown
# And train it the same way:
# %% codecell
data = ml_datasets.mnist()
optimizer = Adam(0.001)
wrapped_mx_model.initialize(X=train_X[:5], Y=train_Y[:5])
train_model(data, wrapped_mx_model, optimizer, n_iter=10, batch_size=128)
# %% markdown
# ---
#
# ## Documentation and resources
#
# - <kbd>USAGE</kbd> [Configuration files](https://thinc.ai/docs/usage-config)
# - <kbd>USAGE</kbd> [Defining and using models](https://thinc.ai/docs/usage-models)
# - <kbd>USAGE</kbd> [Using Thinc with PyTorch, TensorFlow & MXNet](https://thinc.ai/docs/usage-frameworks)
# - <kbd>API</kbd> [Available layers and combinators](https://thinc.ai/docs/api-layers)
# - <kbd>API</kbd> [`Config` and `registry`](https://thinc.ai/docs/api-config)
# - <kbd>API</kbd> [`Model` class](https://thinc.ai/docs/api-model)
