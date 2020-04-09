# %% markdown [markdown]
# [Blog Source](https://synergo.atlassian.net/wiki/spaces/DataScience/pages/1511359082/Building+the+Transformer+XL+from+Scratch)
# $\hspace{1em}$ | $\hspace{1em}$
# [Code Source](https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/transformer_xl_from_scratch.ipynb)
# # Building the [Transformer XL](https://synergo.atlassian.net/wiki/spaces/KnowRes/pages/1513586716) from Sratch
# %% codecell
import numpy as np

import torch
import torch.nn as nn
import torch.tensor as Tensor
from torch import Size, Tensor
from torch.nn.parameter import Parameter
from torch.nn import Dropout, LayerNorm, Linear, Sequential, ReLU, Embedding, ModuleList, CrossEntropyLoss

import matplotlib.pyplot as plt
import sys
import os
from IPython.display import Image

from typing import *

# Making files in utils folder visible here: to import my local print functions for nn.Module objects
sys.path.append(os.getcwd() + "/src/utils/")
from src.utils.ModelUtil import *

# Preparing to show images:
# import ImageResizer

# Building pathname for images
# Set current working directory
os.chdir('/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP')
imagePath = os.getcwd() # now path is the above
print(f"imagePath = {imagePath}\n")
imagePath += "/src/ModelStudy/images/"
print(f"imagePath = {imagePath}")


# %% markdown [markdown]
# # Training the Transformer XL
# %% codecell
TESTING: bool = True


N = 1000
L = 4 # num layers
M = 5 # memory length
H = 4 # num heads
S = 7 # sequence length (sentence length)
P = 6 # previous sequence length
B = 3 # batch size
E = 32 # embedding dimension
I, F = 17, 71 # mhaInnerDim, ffInnerDim

# %% markdown [markdown]
# ### Train Step 1: Prepare Configurations
# The configurations we will be using:
# %% codecell
from src.ModelStudy.TransformerXL.Config import Config


# We will use prime numbers as a dummy test to ensure our implementation is correct
config: Config = Config(seed = 101, debug = False, warmupStep = 0,
                        minLearnRate = 0., # Check default params:
                        dropoutA = 0., # dropout for attention
                        clip = 0.25,
                        logInterval = 200,
                        evalInterval = 100)
config
# %% codecell
if TESTING:
    config.update(fromDict = dict(
        debug = True,
        learningRate = 0.00025,
        batchSize = 8, # batch size
        numEpochs = 2,
        maxStep = 10000, # shorten for testing
        numLayers = L, # 4
        numHeads = H, # 3
        modelDim = E, # 32
        mhaInnerDim = I, # 17
        ffInnerDim = 71,
        dropoutO = 0.1,
        trainBPTT = 33,
        evalBPTT = 41,
        memoryLen = 41,
        evalMemoryLen = 63
    ))
else:
    config.update(fromDict = dict(
        #debug = True,
        learningRate = 0.00025,
        batchSize = 22, # batch size
        numEpochs = 2,
        maxStep = 400000, # shorten for testing
        numLayers = 12,
        numHeads = 8,
        modelDim = 512,
        mhaInnerDim = 64,
        ffInnerDim = 2048,
        dropoutO = 0.1,
        trainBPTT = 512,
        evalBPTT = 128,
        memoryLen = 512,
        evalMemoryLen = 2100
    ))

config


# %% markdown [markdown]
# ### Train Step 2: Preparing the Data Loader
# Data loading for the Transformer Xl is similar to data loading for an RNN based language model but is different from standard data loading.
#
# **Data Loading for Transformer XL:** Suppose we chunked the input into sequence of `batchSize = 4` words to feed into the model. Remember that Transformer XL is stateful, meaning the computations of each minibatch are carried over to the next minibatch. ($\color{red}{\text{Question: is this referring to how } \texttt{newMemory } \text{is computed in the } \texttt{forward } \text{method of the } \texttt{TransformerXL} \text{class?}}$). For a minibatch of size `batchSize = 1`, handling this is simple. We just chunk the input and feed it into the model like this:
# %% codecell
Image(filename =imagePath + "batchsizeone_wrong.png")
# %% markdown [markdown]
# Now what happens if the `batchSize = 2`? We can't split the sentence like this (below) otherwise we would be breaking the dependencies between segments:
# %% codecell
Image(filename =imagePath + "batchsizetwo_wrong.png")
# %% markdown [markdown]
# The correct way to split the corpus with `batchSize = 2` is to feed the batches like this (below). We should have the sentences split across batches rather than keeping as much of the sentence within the batch, and letting the rest of the sentence split across the rest of the batch.
# %% codecell
Image(filename =imagePath + "batchsizetwo_correct.png")

# %% markdown [markdown]
# **General Rule:** Generalizing this, we first divide the corpus into `batchSize` length segments, then feed each segment piece by piece into the model.
#
# **Example of Batching and Feeding:** Suppose `batchSize = 4` and our entire corpus looks like this:
#
# `pytorch is an amazing deep learning framework that makes nlp really easy`
#
# We want to ensure the previous batch contains the previous segment at the same position. In other words, assuming we fed the model one word at a time, we want to iterate over this sentence like this:
#
# `Batch 1: pytorch  amazing   framework  nlp
# Batch 2: is       deep      that       really
# Batch 3: an       learning  makes      easy`
#
# **Key feature of the Method:** We can reconstruct the original sentence by reading from  **top to bottom -> left to right** instead of **left to right -> top to bottom**. Basically we create batches by splitting the sentence *across* batch structure not *within* batch structure.
#
# In reality, we feed the model with a sequence of words for each batch. The length of this sequence is commonly referred to the `bptt` (back propagation through time) length, since this is the maximum length the gradients propagate through in the sequence direction. With a longer `bptt` length of 2 for example, the `minibatch` would be of shape `(batchSize, bptt)` and would look like:
#
# `Batch 1: pytorch  amazing   framework  nlp
#          is       deep      that       really
# Batch 2: an       learning  makes      easy`
#
# We can implement this in a `DataLoader` like this:


# %% codecell
from src.ModelStudy.TransformerXL.LMDataLoader import LMDataLoader


# Testing out the data loader implementation
(N, B, BPTT) = (1000, 16, 10)
testCorpus: Tensor = torch.arange(N)
testCorpus[:BPTT]
# %% codecell
loader: LMDataLoader = LMDataLoader(data = testCorpus, batchSize = B, bptt = BPTT)

loaderIter: List[Tuple[Tensor, Tensor, int]] = list(iter(loader))
batch_0, target_0, diff_0 = loaderIter[0]
batch_0
# %% codecell
target_0

assert (batch_0 + 1 == target_0).all(), "Test target values are shifted one higher than values in batch"
# %% codecell
assert (batch_0 == batch_0[0:BPTT, :]).all(), "Test first dimension of batch has length BPTT"
assert batch_0.shape == (BPTT, B) == (10, 16)
# %% codecell
allBatches: List[Tensor] = [b for b,_,_ in loaderIter]
allTargets: List[Tensor] = [t for _,t,_ in loaderIter]
allDiffs: List[Tensor] = [d for _,_,d in loaderIter]

BatchTensor = Tensor
BPTTTensor = Tensor

def getBPTTCols(colIndex: int, tensors: List[BatchTensor]) -> List[BPTTTensor]:
    """Expects the elements in tensors list to have shape == (BPTT, B) so that when indexing along columns, it gets a list of tensors which are all shape == (BPTT, )
    """
    return [tensors[i][:,colIndex] for i in range(0, len(tensors))]

getBPTTCols(0, allBatches)

# %% codecell
getBPTTCols(0, allTargets)
# %% codecell
getBPTTCols(1, allBatches)
# %% codecell
getBPTTCols(2, allBatches)


# %% markdown [markdown]
# ### Train Step 3: Loading the Actual Data
# Using the Penn Treebank dataset to benchmark our model:
# %% codecell
from pathlib import Path
DATASET_NAME_STR: str = "penn"

# os.chdir('/development/projects/statisticallyfit/github/learningmathstat/PythonNeuralNetNLP')
dataPath: str = os.getcwd() + "/src/ModelStudy/TransformerXL/data/"
DATA_DIR: Path = Path(dataPath) / DATASET_NAME_STR
DATA_DIR.absolute()

# %% markdown [markdown]
# Using a utility vocabulary class borrowed directly from the Transformer XL repo to numericalize our inputs.
# %% codecell
# sys.path.append(os.getcwd() + "/src/ModelStudy/TransformerXL/utils/")
sys.path.append(os.getcwd() + "/src/ModelStudy/TransformerXL/")
# sys.path.pop()
sys.path

# %% codecell
from src.ModelStudy.TransformerXL.vocabulary import Vocab

vocab: Vocab = Vocab(special = ["<eos>"], lower_case = True)

assert (DATA_DIR / "train.txt").absolute() == (DATA_DIR / "train.txt")

(DATA_DIR / "train.txt").absolute()

# %% codecell
trainVocab: List[List[str]] = vocab.count_file(DATA_DIR / "train.txt")
# The `Counter` object in `vocab` counts how many times the token / word has appeared (cumulatively for all the text)
print(list(vocab.counter.items())[:20]) # some token counts
# %% codecell
validVocab: List[List[str]] = vocab.count_file(DATA_DIR / "valid.txt")
print(list(vocab.counter.items())[:20]) # validation text has added 10 'aer' tokens, for instance
# %% codecell
testVocab: List[List[str]] = vocab.count_file(DATA_DIR / "test.txt")
print(list(vocab.counter.items())[:20]) # some token counts
# %% codecell
print(f"trainVocab length = {len(trainVocab)}")
print(f"validVocab length = {len(validVocab)}")
print(f"testVocab length = {len(testVocab)}")

# %% codecell
print(trainVocab[3000:3010])
# %% codecell
print(validVocab[3000:3010])
# %% codecell
print(testVocab[3000:3010])
# %% codecell
lengthsTrain: List[int] = [len(tokenList) for tokenList in trainVocab]
print(f"lengthsTrain[4000:4200]: \n\n{lengthsTrain[4000:4200]}\n")

lengthsValid: List[int] = [len(tokenList) for tokenList in validVocab]
print(f"lengthsValid[2000:2200]: \n\n{lengthsValid[2000:2200]}\n")

lengthsTest: List[int] = [len(tokenList) for tokenList in testVocab]
print(f"lengthsTest[2000:2200]: \n\n{lengthsTest[2000:2200]}\n")


# %% markdown
# ### Train Step 4: Build the Vocabulary
# Encoding the vocabulary text `List[List[str]]` into Tensor of numbers.
# %% codecell
vocab.build_vocab()
# %% markdown
# Encoding the text into tensors:
# %% codecell
ADD_EOS, ADD_DOUBLE_EOS = True, False
trainData: Tensor = vocab.encode_file(path = DATA_DIR / "train.txt", ordered = True, add_eos = ADD_EOS, add_double_eos = ADD_DOUBLE_EOS, verbose = True)
trainData.shape
# %% codecell
trainData
# %% markdown
# Illustrating for the first line  how tokenization and encoding occur:
# %% codecell
print(trainVocab[0])
# %% codecell
line_0: str = ' '.join(trainVocab[0])
line_0
# %% codecell
symbols_0: List[str] = vocab.tokenize(line = line_0, add_eos = ADD_EOS, add_double_eos = ADD_DOUBLE_EOS)
print(symbols_0)
# %% codecell
tensor_0: Tensor = vocab.convert_to_tensor(symbols = symbols_0)
tensor_0

assert (tensor_0 == trainData[0:25]).all()


# %% codecell
validData: Tensor = vocab.encode_file(path = DATA_DIR / "valid.txt", ordered = True, add_eos = True, add_double_eos = False, verbose = True)
validData
# %% codecell
testData: Tensor = vocab.encode_file(path = DATA_DIR / "test.txt", ordered = True, add_eos = True, add_double_eos = False, verbose = True)
testData

# %% markdown
# ### Train Step 5: Prepare the Data Loaders
# %% codecell
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")


trainIter: LMDataLoader = LMDataLoader(data = trainData,
                                       batchSize = config.batchSize,
                                       bptt = config.trainBPTT,
                                       device = device)

validIter: LMDataLoader = LMDataLoader(data = validData,
                                       batchSize = config.batchSize,
                                       bptt = config.trainBPTT,
                                       device = device)

testIter: LMDataLoader = LMDataLoader(data = testData,
                                      batchSize = config.batchSize,
                                      bptt = config.trainBPTT,
                                      device = device)

# %% markdown
# Checking the sizes and shapes of the `batch`, `target`, and `diff` in the `trainLoaderIter`.
# %% codecell
# Studying the shapes for trainIter loader:

# This is the output of the __iter__() method in LMDataLoader
trainLoaderIter: List[Tuple[Tensor, Tensor, int]] = list(iter(trainIter))


for batch, target, diff in trainLoaderIter[: len(trainLoaderIter) - 1]:
    assert batch.names == target.names == ('S', 'B')
    assert batch.shape == target.shape == (config.trainBPTT, config.batchSize)
    assert (batch == batch[0 : config.trainBPTT, :]).all()
    assert diff == config.trainBPTT

batchTargShapes = [(batch.shape, target.shape , diff) for batch, target, diff in trainLoaderIter]

# Last one is the remainder sizes:
print(batchTargShapes[:5], "\n...\n", batchTargShapes[3520:3522]) # note: last one is different!

# %% codecell
allBatches: List[Tensor] = [b for b,_,_ in trainLoaderIter]
allTargets: List[Tensor] = [t for _,t,_ in trainLoaderIter]
allDiffs: List[Tensor] = [d for _,_,d in trainLoaderIter]



# %% markdown
# Visualizing the selected BPTT-length columns from selected batches:
# %% codecell
(BATCH_TUPLE_POS, TARG_TUPLE_POS) = (0, 1)

ITER: int = 0
COL_ID: int = 0

assert (allBatches[ITER] == trainLoaderIter[ITER][BATCH_TUPLE_POS]).all()

print(f"Column ID = {COL_ID} of batch = {ITER} in allBatches: \n\n {getBPTTCols(COL_ID, allBatches)[ITER]}\n\n")

print(f"Batch ID = {ITER}:\n\n {allBatches[ITER]}")

# %% codecell
(BATCH_TUPLE_POS, TARG_TUPLE_POS) = (0, 1)

ITER: int = 1
COL_ID: int = 0

assert (allBatches[ITER] == trainLoaderIter[ITER][BATCH_TUPLE_POS]).all()

print(f"Column ID = {COL_ID} of batch = {ITER} in allBatches: \n\n {getBPTTCols(COL_ID, allBatches)[ITER]}\n\n")

print(f"Batch ID = {ITER}:\n\n {allBatches[ITER]}")

# %% codecell
(BATCH_TUPLE_POS, TARG_TUPLE_POS) = (0, 1)

ITER: int = 2
COL_ID: int = 3

assert (allBatches[ITER] == trainLoaderIter[ITER][BATCH_TUPLE_POS]).all()

print(f"Column ID = {COL_ID} of batch = {ITER} in allBatches: \n\n {getBPTTCols(COL_ID, allBatches)[ITER]}\n\n")

print(f"Batch ID = {ITER}:\n\n {allBatches[ITER]}")
# %% codecell
(BATCH_TUPLE_POS, TARG_TUPLE_POS) = (0, 1)

ITER: int = 4
COL_ID: int = 5

assert (allBatches[ITER] == trainLoaderIter[ITER][BATCH_TUPLE_POS]).all()

print(f"Column ID = {COL_ID} of batch = {ITER} in allBatches: \n\n {getBPTTCols(COL_ID, allBatches)[ITER]}\n\n")

print(f"Batch ID = {ITER}:\n\n {allBatches[ITER]}")

# %% codecell
(BATCH_TUPLE_POS, TARG_TUPLE_POS) = (0, 1)

ITER: int = 10
COL_ID: int = config.batchSize - 1

assert (allBatches[ITER] == trainLoaderIter[ITER][BATCH_TUPLE_POS]).all()

print(f"Column ID = {COL_ID} of batch = {ITER} in allBatches: \n\n {getBPTTCols(COL_ID, allBatches)[ITER]}\n\n")

print(f"Batch ID = {ITER}:\n\n {allBatches[ITER]}")


# %% codecell
(BATCH_TUPLE_POS, TARG_TUPLE_POS) = (0, 1)

ITER: int = 2
COL_ID: int = 3

assert (allTargets[ITER] == trainLoaderIter[ITER][TARG_TUPLE_POS]).all()

print(f"Column ID = {COL_ID} of target = {ITER} in allTargets: \n\n {getBPTTCols(COL_ID, allTargets)[ITER]}\n\n")

print(f"Batch ID = {ITER}:\n\n {allTargets[ITER]}")
# %% markdown
# ### Train Step 6: Initialization
# Initializing the weights and biases, [borrowing the implementation from the Transformer XL repo](https://github.com/kimiyoung/transformer-xl/blob/81b1b1955b5729b311e1548998eb2a89cb528178/pytorch/train.py#L207-L256):
# %% codecell
def initWeight(weight: Tensor) -> Tensor:
    # TODO shape of this tensor??
    # Sets a value INSIDE the given argument, and also RETURNS its value at the same time
    # so this is not good data mutability principle!!!! (bad python)
    nn.init.normal_(tensor = weight, mean = 0.0, std = 0.02)


def initBias(bias: Tensor) -> Tensor:
    # Fills tensor `bias` with value `val`
    nn.init.constant_(tensor = bias, val = 0.0)

def moduleWeightsInit(module: nn.Module) -> Tensor:
    classname: str = module.__class__.__name__

    if classname.find('Linear') != -1:
        if hasattr(module, 'weight') and module.weight is not None:
            initWeight(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            initBias(module.bias)

    elif classname.find('Embedding') != -1:
        if hasattr(module, 'weight'):
            initWeight(module.weight)

    elif classname.find('LayerNorm') != -1:
        if hasattr(module, 'weight'):
        # Fill the argument tensor with normal random values with mu = 1, sigma = 0.02
            nn.init.normal_(tensor = module.weight, mean = 1.0, std = 0.02)
        if hasattr(module, 'bias') and module.bias is not None:
            initBias(module.bias)

    else:
        if hasattr(module, 'u'):
            initWeight(module.u)
        if hasattr(module, 'v'):
            initWeight(module.v)


# %% markdown
# ### Train Step 7: Training Loop
# Training loop is standard, going to write our own to simplify things, but could use ignite, allennlp, fastai.
# %% codecell
# import torch.optim as optim
import torch.optim as optim #
# from torch.optim.optimizer import Optimizer
import math
import time
#import os
from tqdm import tqdm

from torch.utils.data import DataLoader

from src.ModelStudy.TransformerXL.TransformerXL import  TransformerXL

# %% markdown
# Language models are usually evaluated by perplexity.
#
# **Definition: Perplexity:**
#
# Perplexity is the exponential of the cross entropy loss,
# and is also equivalent to the reciprocal of the likelihood. If the language model assigns a probability of $0.1$ to
# each word in the input sentence on average, it would receive a perplexity of $100$.
#
# Intuitively, perplexity represents how many tries it would take for the model to guess the correct word. A
# perplexity of $100$ signifies the model would need $100$ tries to guess each word in the input sequence correctly.
#
# Then the evaluation code becomes:
# %% codecell


def evaluate(model: TransformerXL, validLoader: DataLoader) -> float:

    # Turn on evaluation mode which disables dropout.
    model.eval()

    model.resetLength(seqLen = config.evalBPTT,
                      extLen = 0,
                      memoryLen=config.evalMemoryLen + config.trainBPTT - config.evalBPTT)

    # Evaluation
    totalLen, totalLoss = 0, 0.0

    with torch.no_grad():
        memories: List[Tensor] = None

        for i, (data, target, seqLen) in enumerate(validLoader):
            outDict: Dict[str, Tensor] = model(data, target, memory = memories) # TODO type

            resultLoss: Tensor = outDict["loss"] # tensor of single number
            resultMemories: List[Tensor] = outDict["memory"] # list of tensor memories

            totalLoss += seqLen * resultLoss.float().item()
            totalLen += seqLen

    # Switch back to the training mode (to maintain state since this evaluate() function is used in the training loop
    # so we need to set the state back to training state)
    model.resetLength(seqLen = config.trainBPTT,
                      extLen = 0,
                      memoryLen = config.memoryLen)
    model.train()

    return totalLoss / totalLen

# %% codecell

trainLossChange: List[Tensor] = []
validLossChange = [] # TODO type


def trainEpoch(numEpoch: int,
               model: TransformerXL,
               trainLoader: DataLoader, validLoader: DataLoader,
               optimizer: optim.Optimizer,
               scheduler,
               trainStepStart: float = 0.):

    # Turn on training mode which enables dropout
    model.train()
    memories: List[Tensor] = None
    trainStep: float = trainStepStart
    trainLoss: float = 0

    # Time-book-keeping
    loggerStartTime: float = time.time()

    bestValidationLoss: float = float('inf')

    progressBar = tqdm(trainLoader,
                       total = min(config.maxStep - trainStepStart, len(trainLoader)))


    for iBatch, (data, target, seqLen) in enumerate(progressBar):
        model.zero_grad()

        outDict: Dict[str, Tensor] = model(data, target, memory = memories)
        resultLoss: Tensor = outDict["loss"] # tensor of single number
        resultMemories: List[Tensor] = outDict["memory"] # list of tensor memories

        resultLoss.backward()
        trainLoss += resultLoss.item()
        trainLossChange.append(resultLoss.item())

        torch.nn.utils.clip_grad_norm_(parameters = model.parameters(),
                                       max_norm = config.clip)

        optimizer.step()

        # Step-wise learning rate annlealing
        trainStep += 1

        # Linear warm up stage
        if trainStep < config.warmupStep:
            currLearnRate = config.learningRate * trainStep / config.warmupStep
            optimizer.param_groups[0]['lr'] = currLearnRate

        else:
            scheduler.step(trainStep)


        # LOGGING updates
        if trainStep % config.logInterval == 0:
            currLoss: float = trainLoss / config.logInterval

            elapsedTime: float = time.time() - loggerStartTime

            loggerTimeStr: str = '| epoch {:3d} step {:>8d} | lr {:.3g} ' \
                                 '| loss {:5.2f}'.format(
                numEpoch, trainStep, optimizer.param_groups[0]['lr'], currLoss)

            loggerTimeStr += ' | PPL (perplexity) {:9.3f}'.format(math.exp(currLoss))

            progressBar.set_description(desc = loggerTimeStr)

            trainLoss = 0 # reset the training loss after reporting it

            loggerStartTime = time.time() # start again for this point on

        # EVALUATION updates
        if trainStep % config.evalInterval == 0:
            validLoss: float = evaluate(model, validLoader)
            validLossChange.append(validLoss)

            evaluationStartTime = time.time()

        if trainStep == config.maxStep:
            return trainStep

    return trainStep


# %% codecell

# TODO argument and return types ...
def train(model: TransformerXL, trainLoader: DataLoader, validLoader: DataLoader):

    optimizer: optim.Optimizer = optim.Adam(params = model.parameters(),
                                            lr = config.learningRate)

    numTotalSteps: int = min(config.maxStep, len(trainLoader) * config.numEpochs)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                     T_max = numTotalSteps,
                                                     eta_min = config.minLearnRate)
    trainStepStart: int = 0

    for numEpoch in range(config.numEpochs):
        if trainStepStart >= config.maxStep:
            break

        trainStepStart = trainEpoch(numEpoch = numEpoch,
                                    model = model ,
                                    trainLoader = trainLoader,
                                    validLoader = validLoader,
                                    optimizer = optimizer,
                                    scheduler = scheduler,
                                    trainStepStart = trainStepStart)




# %% codecell
def evaluateFinal(model: TransformerXL,
                  validLoader: DataLoader) -> Dict[str, float]:

    # Set to evaluation mode
    model.eval()
    model.resetLength(seqLen = config.evalBPTT,
                      extLen = 0,
                      memoryLen=config.evalMemoryLen + config.trainBPTT - config.evalBPTT)

    # Evaluation
    totalLen, totalLoss = 0, 0.0

    evalStartTime: float = time.time()

    with torch.no_grad():
        memories: List[Tensor] = None

        for i, (data, target, seqLen) in enumerate(validLoader):
            outDict: Dict[str, Tensor] = model(data, target, memory = memories)

            resultLoss: Tensor = outDict["loss"] # tensor of single number
            resultMemories: List[Tensor] = outDict["memory"] # list of tensor memories

            totalLoss += seqLen * resultLoss.item() # item inside tensor
            totalLen += seqLen

        elapsedTime = time.time() - evalStartTime

    # TODO setting back to train mode? if so then don't we need `model.train()` like in evaluate()???
    model.resetLength(seqLen = config.trainBPTT,
                      extLen = 0,
                      memoryLen = config.memoryLen)

    validLoss: float = totalLoss / totalLen

    return {"validationLoss": validLoss, "PPL": math.exp(validLoss)}



# %% markdown
# ### Train Step 8: Train the Model!
# Now all we have to do is initialize the model and start training it - actually!

# %% codecell
transformerXLToTrain: TransformerXL = TransformerXL(
    numEmbeddings = len(vocab),
    numLayers = config.numLayers,
    numHeads = config.numHeads,
    modelDim = config.modelDim,  # E
    mhaInnerDim = config.mhaInnerDim, # I
    ffInnerDim = config.ffInnerDim,  # F
    dropoutO = config.dropoutO,
    dropoutA = config.dropoutA,
    seqLen = config.trainBPTT, # S
    memoryLen = config.memoryLen # M
)

if torch.cuda.is_available():
    transformerXLToTrain.cuda()

transformerXLToTrain.apply(fn = moduleWeightsInit)
# %% codecell
briefParams(transformerXLToTrain)

# %% codecell
train(model = transformerXLToTrain,
      trainLoader = trainIter,
      validLoader = validIter
      )

# %% markdown
# Now evaluating:
# %% codecell
resultDict: Dict[str, float] = evaluateFinal(model = transformerXLToTrain, validLoader = validIter)
resultDict
# %% markdown
# ### Visualizing: Loss Change
# Overall the loss is decreasing - both the `lossChange` and `validLossChange`
# %% codecell
import matplotlib.pyplot as plot
# %matplotlib inline

plt.plot(trainLossChange)
# %% codecell
plt.plot(validLossChange)
