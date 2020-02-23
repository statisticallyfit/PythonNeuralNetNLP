# %% markdown
# # Training a part-of-speech tagger with transformers (BERT)
#
# This example shows how to use Thinc and Hugging Face's [`transformers`](https://github.com/huggingface/transformers) library to implement and train a part-of-speech tagger on the Universal Dependencies [AnCora corpus](https://github.com/UniversalDependencies/UD_Spanish-AnCora). This notebook assumes familiarity with machine learning concepts, transformer models and Thinc's config system and `Model` API (see the "Thinc for beginners" notebook and the [documentation](https://thinc.ai/docs) for more info).
# %% codecell
!pip install "thinc>=8.0.0a0" transformers torch ml_datasets "tqdm>=4.41"
# %% markdown
# First, let's use Thinc's `prefer_gpu` helper to make sure we're performing operations **on GPU if available**. The function should be called right after importing Thinc, and it returns a boolean indicating whether the GPU has been activated. If we're on GPU, we can also call `use_pytorch_for_gpu_memory` to route `cupy`'s memory allocation via PyTorch, so both can play together nicely.
# %% codecell
from thinc.api import prefer_gpu, use_pytorch_for_gpu_memory

is_gpu = prefer_gpu()
print("GPU:", is_gpu)
if is_gpu:
    use_pytorch_for_gpu_memory()
# %% markdown
# ## Overview: the final config
#
# Here's the final config for the model we're building in this notebook. It references a custom `TransformersTagger` that takes the name of a starter (the pretrained model to use), an optimizer, a learning rate schedule with warm-up and the general training settings. You can keep the config string within your file or notebook, or save it to a `conig.cfg` file and load it in via `Config.from_disk`.
# %% codecell
CONFIG = """
[model]
@layers = "TransformersTagger.v1"
starter = "bert-base-multilingual-cased"

[optimizer]
@optimizers = "Adam.v1"

[optimizer.learn_rate]
@schedules = "warmup_linear.v1"
initial_rate = 0.01
warmup_steps = 3000
total_steps = 6000

[loss]
@losses = "SequenceCategoricalCrossentropy.v1"

[training]
batch_size = 128
words_per_subbatch = 2000
n_epoch = 10
"""
# %% markdown
# ---
#
# ## Defining the model
#
# The Thinc model we want to define should consist of 3 components: the transformers **tokenizer**, the actual **transformer** implemented in PyTorch and a **softmax-activated output layer**.
#
#
# ### 1. Wrapping the tokenizer
#
# To make it easier to keep track of the data that's passed around (and get type errors if something goes wrong), we first create a `TokensPlus` dataclass that holds the output of the `batch_encode_plus` method of the `transformers` tokenizer. You don't _have to_ do this, but it makes things easier, can prevent bugs and helps the type checker.
# %% codecell
from typing import Optional, List
from dataclasses import dataclass
import torch

@dataclass
class TokensPlus:
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    input_len: List[int]
    overflowing_tokens: Optional[torch.Tensor] = None
    num_truncated_tokens: Optional[torch.Tensor] = None
    special_tokens_mask: Optional[torch.Tensor] = None
# %% markdown
# The wrapped tokenizer will take a list-of-lists as input (the texts) and will output a `TokensPlus` object containing the fully padded batch of tokens. The wrapped transformer will take a list of `TokensPlus` objects and will output a list of 2-dimensional arrays.
#
# 1. **TransformersTokenizer**: `List[List[str]]` → `TokensPlus`
# 2. **Transformer**: `TokensPlus` → `List[Array2d]`
#
# > 💡 Since we're adding type hints everywhere (and Thinc is fully typed, too), you can run your code through [`mypy`](https://mypy.readthedocs.io/en/stable/) to find type errors and inconsistencies. If you're using an editor like Visual Studio Code, you can enable `mypy` linting and type errors will be highlighted in real time as you write code.
#
# To wrap the tokenizer, we register a new function that returns a Thinc `Model`. The function takes the name of the pretrained weights (e.g. `"bert-base-multilingual-cased"`) as an argument that can later be provided via the config. After loading the `AutoTokenizer`, we can stash it in the attributes. This lets us access it at any point later on via `model.attrs["tokenizer"]`.
# %% codecell
import thinc
from thinc.api import Model
from transformers import AutoTokenizer

@thinc.registry.layers("transformers_tokenizer.v1")
def TransformersTokenizer(name: str) -> Model[List[List[str]], TokensPlus]:
    def forward(model, texts: List[List[str]], is_train: bool):
        tokenizer = model.attrs["tokenizer"]
        token_data = tokenizer.batch_encode_plus(
            [(text, None) for text in texts],
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_masks=True,
            return_input_lengths=True,
            return_tensors="pt",
        )
        return TokensPlus(**token_data), lambda d_tokens: []

    return Model("tokenizer", forward, attrs={"tokenizer": AutoTokenizer.from_pretrained(name)})
# %% markdown
# The forward pass takes the model and a list-of-lists of strings and outputs the `TokensPlus` dataclass and a callback to use during the backwards (which does nothing in this case).
# %% markdown
# ### 2. Wrapping the transformer
#
# To load and wrap the transformer, we can use `transformers.AutoModel` and Thinc's `PyTorchWrapper`. The forward method of the wrapped model can take arbitrary positional arguments and keyword arguments. Here's what the wrapped model is going to look like:
#
# ```python
# @thinc.registry.layers("transformers_model.v1")
# def Transformer(name) -> Model[TokensPlus, List[Array2d]]:
#     return PyTorchWrapper(
#         AutoModel.from_pretrained(name),
#         convert_inputs=convert_transformer_inputs,
#         convert_outputs=convert_transformer_outputs,
#     )
# ```
#
# The transformer takes `TokensPlus` data as input (as produced by the tokenizer) and outputs a list of 2-dimensional arrays. The convert functions are used to **map inputs and outputs to and from the PyTorch model**. Each function should return the converted output, and a callback to use during the backward pass. To make the arbitrary positional and keyword arguments easier to manage, Thinc uses an `ArgsKwargs` dataclass, essentially a named tuple with `args` and `kwargs` that can be spread into a function as `*ArgsKwargs.args` and `**ArgsKwargs.kwargs`. The `ArgsKwargs` objects will be passed straight into the model in the forward pass, and straight into `torch.autograd.backward` during the backward pass.
# %% codecell
from thinc.api import ArgsKwargs, torch2xp, xp2torch
from thinc.types import Array2d

def convert_transformer_inputs(model, tokens: TokensPlus, is_train):
    kwargs = {
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask,
        "token_type_ids": tokens.token_type_ids,
    }
    return ArgsKwargs(args=(), kwargs=kwargs), lambda dX: []


def convert_transformer_outputs(model, inputs_outputs, is_train):
    layer_inputs, torch_outputs = inputs_outputs
    torch_tokvecs: torch.Tensor = torch_outputs[0]
    torch_outputs = None  # free the memory as soon as we can
    lengths = list(layer_inputs.input_len)
    tokvecs: List[Array2d] = model.ops.unpad(torch2xp(torch_tokvecs), lengths)
    tokvecs = [arr[1:-1] for arr in tokvecs]  # remove the BOS and EOS markers

    def backprop(d_tokvecs: List[Array2d]) -> ArgsKwargs:
        # Restore entries for BOS and EOS markers
        row = model.ops.alloc2f(1, d_tokvecs[0].shape[1])
        d_tokvecs = [model.ops.xp.vstack((row, arr, row)) for arr in d_tokvecs]
        return ArgsKwargs(
            args=(torch_tokvecs,),
            kwargs={"grad_tensors": xp2torch(model.ops.pad(d_tokvecs))},
        )

    return tokvecs, backprop
# %% markdown
# The model returned by `AutoModel.from_pretrained` is a PyTorch model we can wrap with Thinc's `PyTorchWrapper`. The converter functions tell Thinc how to transform the inputs and outputs.
# %% codecell
import thinc
from thinc.api import PyTorchWrapper
from transformers import AutoModel

@thinc.registry.layers("transformers_model.v1")
def Transformer(name: str) -> Model[TokensPlus, List[Array2d]]:
    return PyTorchWrapper(
        AutoModel.from_pretrained(name),
        convert_inputs=convert_transformer_inputs,
        convert_outputs=convert_transformer_outputs,
    )
# %% markdown
# We can now combine the `TransformersTokenizer` and `Transformer` into a feed-forward network using the `chain` combinator. The `with_array` layer transforms a sequence of data into a contiguous 2d array on the way into and
# out of a model.
# %% codecell
from thinc.api import chain, with_array, Softmax

@thinc.registry.layers("TransformersTagger.v1")
def TransformersTagger(starter: str, n_tags: int = 17) -> Model[List[List[str]], List[Array2d]]:
    return chain(
        TransformersTokenizer(starter),
        Transformer(starter),
        with_array(Softmax(n_tags)),
    )
# %% markdown
# ---
#
# ## Training the model
#
# ### Setting up model and data
#
# Since we've registered all layers via `@thinc.registry.layers`, we can construct the model, its settings and other functions we need from a config (see `CONFIG` above). The result is a config object with a model, an optimizer, a function to calculate the loss and the training settings.
# %% codecell
from thinc.api import Config, registry

C = registry.make_from_config(Config().from_str(CONFIG))
C
# %% codecell
model = C["model"]
optimizer = C["optimizer"]
calculate_loss = C["loss"]
cfg = C["training"]
# %% markdown
# We’ve prepared a separate package [`ml-datasets`](https://github.com/explosion/ml-datasets) with loaders for some common datasets, including the AnCora data. If we're using a GPU, calling `ops.asarray` on the outputs ensures that they're converted to `cupy` arrays (instead of `numpy` arrays). Calling `Model.initialize` with a batch of inputs and outputs allows Thinc to **infer the missing dimensions**.
# %% codecell
import ml_datasets
(train_X, train_Y), (dev_X, dev_Y) = ml_datasets.ud_ancora_pos_tags()

train_Y = list(map(model.ops.asarray, train_Y))  # convert to cupy if needed
dev_Y = list(map(model.ops.asarray, dev_Y))  # convert to cupy if needed

model.initialize(X=train_X[:5], Y=train_Y[:5])
# %% markdown
# ### Helper functions for training and evaluation
#
# Before we can train the model, we also need to set up the following helper functions for batching and evaluation:
#
# * **`minibatch_by_words`:** Group pairs of sequences into minibatches under `max_words` in size, considering padding. The size of a padded batch is the length of its longest sequence multiplied by the number of elements in the batch.
# * **`evaluate_sequences`:** Evaluate the model sequences of two-dimensional arrays and return the score.
# %% codecell
def minibatch_by_words(pairs, max_words):
    pairs = list(zip(*pairs))
    pairs.sort(key=lambda xy: len(xy[0]), reverse=True)
    batch = []
    for X, Y in pairs:
        batch.append((X, Y))
        n_words = max(len(xy[0]) for xy in batch) * len(batch)
        if n_words >= max_words:
            yield batch[:-1]
            batch = [(X, Y)]
    if batch:
        yield batch

def evaluate_sequences(model, Xs: List[Array2d], Ys: List[Array2d], batch_size: int) -> float:
    correct = 0.0
    total = 0.0
    for X, Y in model.ops.multibatch(batch_size, Xs, Ys):
        Yh = model.predict(X)
        for yh, y in zip(Yh, Y):
            correct += (y.argmax(axis=1) == yh.argmax(axis=1)).sum()
            total += y.shape[0]
    return float(correct / total)
# %% markdown
# ### The training loop
#
# Transformers often learn best with **large batch sizes** – larger than fits in GPU memory. But you don't have to backprop the whole batch at once. Here we consider the "logical" batch size (number of examples per update) separately from the physical batch size. For the physical batch size, what we care about is the **number of words** (considering padding too). We also want to sort by length, for efficiency.
#
# At the end of the batch, we **call the optimizer** with the accumulated gradients, and **advance the learning rate schedules**. You might want to evaluate more often than once per epoch – that's up to you.
# %% codecell
from tqdm.notebook import tqdm
from thinc.api import fix_random_seed

fix_random_seed(0)

for epoch in range(cfg["n_epoch"]):
    batches = model.ops.multibatch(cfg["batch_size"], train_X, train_Y, shuffle=True)
    for outer_batch in tqdm(batches, leave=False):
        for batch in minibatch_by_words(outer_batch, cfg["words_per_subbatch"]):
            inputs, truths = zip(*batch)
            guesses, backprop = model(inputs, is_train=True)
            backprop(calculate_loss.get_grad(guesses, truths))
        model.finish_update(optimizer)
        optimizer.step_schedules()
    score = evaluate_sequences(model, dev_X, dev_Y, cfg["batch_size"])
    print(epoch, f"{score:.3f}")
# %% markdown
# If you like, you can call `model.to_disk` or `model.to_bytes` to save the model weights to a directory or a bytestring.
