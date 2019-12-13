
# %% markdown
# ## Preparing the Data
#
# First, let's import all the required modules and set the random seeds for reproducability.
# %% codecell
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.tensor as Tensor

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# %matplotlib inline
import seaborn as sns

import spacy

import os
import random
import math
import time




# %% codecell
# Set random seeds for reproducibility

SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# %% markdown
# ### 1. Create the Tokenizers
#
# Next, we'll create the tokenizers. A tokenizer is used to turn a string containing a sentence into a list of individual tokens that make up that string, e.g. "good morning!" becomes ["good", "morning", "!"].
#
# spaCy has model for each language ("de" for German and "en" for English) which need to be loaded so we can access the tokenizer of each model.
# %% codecell
# Download the spacy models via command line:
# conda activate pynlp_env
# cd /development/.../NLPStudy/data
# python -m spacy download en
# python -m spacy download de

# Then load the models
#spacyDE = spacy.load('de')

spacyEN = spacy.load('en')
spacyDE = spacy.load('de')
#spacyFR = spacy.load('fr') # french!
#spacyIT = spacy.load('it')
# site link for other language models: https://spacy.io/usage/models
# %% markdown
# ### 2. Create the Tokenizer Functions
#
# Next, we create the tokenizer functions. These can be passed to TorchText and will take in the sentence as a string and return the sentence as a list of tokens.
# %% codecell
# Creating the tokenizer functions
def tokenizeGerman(germanText):
    # tokenizes the german text into a list of strings(tokens) and reverse it
    # we are reversing the input sentences, as it is observed
    # by reversing the inputs we will get better results
    return [tok.text for tok in spacyDE.tokenizer(germanText)][::-1]     # list[::-1] used to reverse the list


def tokenizeEnglish(englishText):
    # tokenizes the english text into a list of strings(tokens)
    return [tok.text for tok in spacyEN.tokenizer(englishText)]

# %% markdown
# We set the tokenize argument to the correct tokenization function for each, with German being the `SRC` (source) field and English being the `TRG` (target) field. The `Field` also appends the "start of sequence" and "end of sequence" tokens via the `init_token` and `eos_token` arguments, and converts all words to lowercase.
#
# [To read more about Field's arguments](https://github.com/pytorch/text/blob/master/torchtext/data/field.py#L61)
#
#  **Difference this time:**
# By default RNN models in PyTorch require the sequence to be a tensor of shape **(`sequenceLength`, `batchSize`)** so TorchText will, by default, return batches of tensors in the same shape.
# To give the Transformer model the batch dimension first, we tell TorchText to have batches be **(`batchSize`, `sequenceLength`)** by setting `batch_first = True`. Then the sequence (sentence) will be returned in batch-major format.
#
# We also append the start and end of sequence tokens as well as lowercasing all text.
#
# German = source language, English = target language

# - `tokenize`:  The function used to tokenize strings using this field into sequential examples. We can configure the method of tokenization using this argument.
# - `init_token`: A token that will be prepended to every example using this field, or `None` for no initial token. Default: `None`.
# - `eos_token`: A token that will be appended to every example using this field, or `None` for no end-of-sentence token. Default: `None`.
# - `lower`: Whether to lowercase the text in this field. Default: `False`.
# %% codecell
SRC: Field = Field(tokenize = tokenizeGerman,
            init_token = '<sos>',   # start of sentence token
            eos_token = '<eos>',    # end of sentence token
            lower = True,           # lowercase all text
            batch_first = True)     # return sequence in batch major format

TRG: Field = Field(tokenize = tokenizeEnglish,
            init_token = '<sos>',   # start of sentence token
            eos_token = '<eos>',    # end of sentence token
            lower = True,           # lowercase all text
            batch_first = True)     # return sequence in batch major format
# %% markdown
# ### 3. Download the Data
#
# Next, we download and load the train, validation and test data using the [Multi30k dataset](https://github.com/multi30k/dataset).
# - `exts` = which language to us as rouce and target, with source specified before target.
# - `fields` = define which data processing to apply to the source and target languages.
# %% codecell
# NOTE: after this, the data is stored in
# a folder under NLPSTUDY called '.data'

trainData, validationData, testData = \
    Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG))

# %% markdown
# Checking the amount of data:
# %% codecell
print(f"Number of training examples: {len(trainData.examples)}")
print(f"Number of validation examples: {len(validationData.examples)}")
print(f"Number of testing examples: {len(testData.examples)}")
# %% markdown
# Print out an example from the training data. See how source (german) is reversed while target (english) is in proper order:
# %% codecell
print(trainData.examples[0])
print(vars(trainData.examples[0]))
# %% markdown
# ### 4. Building the vocabulary
#
# Next, we build the *vocabulary* for the source and target languages because the source and target are in different languages.
#
# The vocabulary is used to associate each unique token (word) with an index (an integer), similar to the way a dictionary works. Torchtext's `Field` creates dicts, maps word to index, maps index to word, counts words, etc.
#
# This is used to build a one-hot encoding for each token (a vector of all zeros except for the position represented by the index, which is 1).
#
# The vocabularies of the source and target languages are distinct.
#
# - `min_freq` = used to allow only tokens that appear a minimum number of times (`min_freq` times) to appear in the vocabulary. If any word appears fewer times, it is not included in the vocabulary. Tokens that appear only once are convered into an `<unk>` (unknown) token.
#
# - WARNING : We will use only training data for creating the vocabulary to prevent information leakage into the model.
# %% codecell
SRC.build_vocab(trainData, min_freq=2)
TRG.build_vocab(trainData, min_freq=2)
# %% markdown
# Checking the size of the `SRC` and `TRG` vocabulary:
# %% codecell
print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
# %% markdown
# ### 5. (Final) Create the Iterators
#
# The final step of preparing the data is to create the iterators.
#
# - Iterators: These can be iterated on to return a batch of data which will have a `src` attribute (the PyTorch tensors containing a batch of numericalized source sentences) and a `trg` attribute (the PyTorch tensors containing a batch of numericalized target sentences). "Numericalized" is just a fancy way of saying they have been converted from a sequence of readable tokens to a sequence of corresponding indexes, using the vocabulary.
# - Indexes: We also need to replace the words by its indexes, since any model takes only numbers as input using the
# `vocabulary`.
# - `device`: Must also define a `torch.device`. This is used to tell TorchText to put the tensors on the GPU or not.
# We use the `torch.cuda.is_available()` function, which will return `True` if a GPU is detected on our computer. We pass this `device` to the iterator.
# - Padding: When we get a batch of examples using an iterator we need to make sure that all of the source sentences
# are padded to the same length, the same with the target sentences. This is handled by the Torchtext iterators.
# - Batching: We use a `BucketIterator` instead of the standard `Iterator` as it creates batches in such a way that it minimizes the amount of padding in both the source and target sentences.

# %% codecell
# use gpu if available, else use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
# %% codecell
# Creating the training iterator

BATCH_SIZE: int = 32         # 128

# Create data iterators for the data
# padding all the sentences to same length, replacing words by its index,
# bucketing (minimizes the amount of padding by grouping similar length sentences)
trainIterator, validationIterator, testIterator = BucketIterator.splits(
    (trainData, validationData, testData),
    batch_size = BATCH_SIZE,
    #sort_within_batch = True, # new key feature
    #sort_key = lambda x: len(x.src), # new key feature
    device = device)




# %% markdown
# ## Training the Seq2Seq Model
#
#
# ### Step 1: Initialize the Seq2Seq Transformer Model

# %% codecell
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HIDDEN_DIM = 512 # d_model
NUM_LAYERS = 6
NUM_HEADS = 8
PFF_DIM = 2048 # poswise feedforward hidden dim
DROPOUT = 0.1
PAD_INDEX = SRC.vocab.stoi['<pad>']


# %% codecell

from src.NLPstudy.TransformerModel.IllustratedTransformer.Encoder import *
from src.NLPstudy.TransformerModel.IllustratedTransformer.EncoderLayer import *
from src.NLPstudy.TransformerModel.IllustratedTransformer.Decoder import *
from src.NLPstudy.TransformerModel.IllustratedTransformer.DecoderLayer import *
from src.NLPstudy.TransformerModel.IllustratedTransformer.Transformer import *
from src.NLPstudy.TransformerModel.IllustratedTransformer.SelfAttentionLayer import *
from src.NLPstudy.TransformerModel.IllustratedTransformer.PositionwiseFeedforwardLayer import *
from src.NLPstudy.TransformerModel.IllustratedTransformer.PositionalEncodingLayer import *




# %% codecell
posEnc = PositionalEncodingLayer(d_model = HIDDEN_DIM, dropout = DROPOUT, device = device)
posEnc


# %% codecell
# TODO: why are classes instead of objects passed here?

encoder: Encoder = Encoder(inputDim = INPUT_DIM, hiddenDim = HIDDEN_DIM, numLayers = NUM_LAYERS,
                           numHeads = NUM_HEADS, pffHiddenDim = PFF_DIM,
                           encoderLayerC= EncoderLayer,
                           attnC= SelfAttentionLayer,
                           pffC= PositionwiseFeedforwardLayer,
                           peC= posEnc,
                           dropout = DROPOUT, device = device)
encoder

# %% codecell
decoder: Decoder = Decoder(outputDim = OUTPUT_DIM, hiddenDim = HIDDEN_DIM, numLayers = NUM_LAYERS,
                           numHeads = NUM_HEADS, pffHiddenDim = PFF_DIM,
                           decoderLayer = DecoderLayer,
                           attnLayer = SelfAttentionLayer,
                           pffLayer = PositionwiseFeedforwardLayer,
                           peLayer = posEnc,
                           dropout = DROPOUT, device = device)
decoder



# %% markdown
# Can see the structure of the transformer model very clearly here:

# %% codecell
transformerModel: Transformer = Transformer(encoder = encoder, decoder = decoder,
                                            padIndex = PAD_INDEX, device = device).to(device)

transformerModel




# %% markdown
# ### Step 2: Initialize Parameter Weights
# %% codecell
def initWeights(model: Transformer):
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

transformerModel.apply(fn = initWeights)
transformerModel

# %% markdown
# ### Step 3: Count Parameters
# %% codecell
def countParameters(model: Transformer):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {countParameters(transformerModel):,} trainable parameters')


# %% markdown
# ### Step 3: Define Optimizer
#
# The paper uses a modified Adam optimizer.
# > *We used the Adam optimizer with $\beta_1 = 0.9$, $\beta_2 = 0.98$, and $\epsilon=10^{-9}$. We varied the learning rate over the course of training, according to the formula: *
# > $$
# > learningRate = d_{model}^{-0.5} \cdot min(stepNum^{-0.5}, stepNum \cdot warmupSteps^{-1.5})
# > $$
# > *This corresponds to increasing the learning rate linearly for the first $warmupSteps$ training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used $warmupSteps = 4000$.*

# %% codecell
# Create the modified adam optimizer class

class NoamOpt:
    "Optimizer wrapper that implements the rate."
    def __init__(self, modelSize: int, factor: float, warmupSteps: int, optimizer):
        self.optimizer = optimizer
        self.currentStep: int = 0
        self.numWarmupSteps: int = warmupSteps
        self.factor: float = factor
        self.modelSize: int = modelSize # d_model
        self.learningRate: float = 0



    def step(self):
        '''Update parameters and the learning rate. '''
        self.currentStep += 1
        currentRate: float = self.calculateLearningRate()

        for par in self.optimizer.param_groups:
            par['lr'] = currentRate

        self.learningRate = currentRate

        self.optimizer.step()


    def calculateLearningRate(self, step = None):
        '''Implement `learningRate` formula above'''
        if step is None:
            step = self.currentStep

        return self.factor * \
               (self.modelSize ** (-0.5) *
                min(step ** (-0.5), step * self.numWarmupSteps ** (-1.5)))



# Create an instance:
adamOptimizer = torch.optim.Adam(params = transformerModel.parameters(),
                                 lr = 0,
                                 betas = (0.9, 0.98),
                                 eps=1e-9)
type(adamOptimizer)

modifAdamOptimizer: NoamOpt = NoamOpt(modelSize = HIDDEN_DIM,
                                      factor = 1,
                                      warmupSteps=2000,
                                      optimizer = adamOptimizer)
modifAdamOptimizer


# %% markdown
# ### Step 4: Define the Loss Function (Cross Entropy)
# The `CrossEntropyLoss` function calculates both log softmax and negative log-likelihood of our predictions.
#
# The loss function calculates the average loss per token but by passing the index of the `<pad>` token as the `ignore_index` argument, we effectively ignore the loss whenever the target token is a padding token.
# %% codecell
crossEntropyLossFunction = nn.CrossEntropyLoss(ignore_index = PAD_INDEX)
crossEntropyLossFunction



# %% markdown
# ### Step 5: Define the Training Loop
#
# Next, we'll define our training loop.
#
# First, we'll set the model into "training mode" with `model.train()`. This will turn on dropout (and batch normalization, which we aren't using) and then iterate through our data iterator.
#
# - **NOTE:**: the `Decoder` loop starts at 1, not 0. This means the 0th element of our `outputs` tensor remains all zeros. So our `trg` and `outputs` look something like:
#
# $$
# \text{trg} = [<sos>, y_1, y_2, y_3, <eos>] \\
# \text{outputs} = [0, \hat{y}_1, \hat{y}_2, \hat{y}_3, <eos>] \\
# $$
#
# Here, when we calculate the loss, we cut off the first element of each tensor to get:
#
# $$
# \text{trg} = [y_1, y_2, y_3, <eos>] \\
# \text{outputs} = [\hat{y}_1, \hat{y}_2, \hat{y}_3, <eos>] \\
# $$
#
# At each iteration:
# - Get the source and target sentences from the batch, $X$ and $Y$
# - Zero the gradients calculated from the last batch
# - Feed the source and target into the model to get the output, $\hat{Y}$
# - As the **loss function only works on 2d inputs with 1d targets we need to flatten each of them with `.view`**
#   - To avoid measuring the loss of the `<sos>` token, we slice off the first column of the output and target tensors as mentioned above.
# - Calculate the gradients with `loss.backward()`
# - Clip the gradients to prevent them from exploding (a common issue in RNNs)
# - Update the parameters of our model by doing an optimizer step
# - Sum the loss value to a running total
#
# Finally, we return the loss that is averaged over all batches.

# %% codecell
def train(model: Transformer, iterator, noamOpt: NoamOpt, lossFunction, clip: int):

    model.train() # put model in training mode

    lossPerEpoch: int = 0

    for epoch, batch in enumerate(iterator):

        # 1. Getting source and target sentences from batch
        srcSentence = batch.src
        trgSentence: Tensor = batch.trg

        # 2. Zero the gradients from the last batch
        noamOpt.optimizer.zero_grad()

        # 3. Feed the source and target sentences into the seq2seq model
        # to get the output tensor of predictions.
        output: Tensor = model(src=srcSentence,
                               trg =trgSentence[:, :-1]) # cutting off first element to avoid passing the
        # <sos>  token
        ### trgSentence = vector of shape (trgSentenceLen * batchSize - 1)
        ### output = tensor of shape (batchSize, trgSentenceLen -1, outputDim)

        # 4. Need to flatten the outputs to be in 2d input with 1d target
        # so that loss can take this as an argument.
        # (by slicing off the first column of the output and target tensors
        # as mentioned above)
        outputDim: int = output.shape[-1]

        output_Reshaped: Tensor = output.contiguous().view(-1, outputDim)
        trgSentence_Reshaped: Tensor = trgSentence[:, 1:].contiguous().view(-1)
        ## trgSentence shape now: (batchSize * trgSentLen - 1)
        ## output shape now: (batchSize * trgSentLen - 1, outputDim)

        # 5. Calculate gradients
        loss = lossFunction(input=output_Reshaped, target= trgSentence_Reshaped)
        loss.backward()

        # 6. Clip gradient so it doesn't explode
        torch.nn.utils.clip_grad_norm_(parameters = model.parameters(),
                                       max_norm = clip)

        # 7. Update parameters of model
        noamOpt.step()

        # 8. Sum the loss value to a running total
        lossPerEpoch += loss.item()

    return lossPerEpoch / len(iterator) # average loss

# %% markdown
# ### Step 7: Define the Evaluation Loop
#
# Our evaluation loop is similar to our training loop, however as we aren't updating any parameters we don't need to pass an optimizer or a clip value.
#
# *We must remember to set the model to evaluation mode with `model.eval()`. This will turn off dropout (and batch normalization, if used).*
#
# We use the `with torch.no_grad()` block to ensure no gradients are calculated within the block. This reduces memory consumption and speeds things up.

# %% codecell
def evaluate(model: Transformer, iterator, lossFunction: nn.CrossEntropyLoss):

    model.eval()

    lossPerEpoch: float = 0

    with torch.no_grad():

        for epoch, batch in enumerate(iterator):

            # 1. Getting source and target sentences from batch
            srcSentence = batch.src
            trgSentence: Tensor = batch.trg

            # 3. Feed the source and target sentences into the seq2seq model
            # to get the output tensor of predictions.
            output: Tensor = model(src=srcSentence,
                                   trg =trgSentence[:, :-1]) # cutting off first element to avoid passing the
            # <sos>  token
            ### trgSentence = vector of shape (batchSize, trgSentenceLen)
            ### output = tensor of shape (batchSize, trgSentenceLen -1, outputDim)


            # 4. Need to flatten the outputs to be in 2d input with 1d target
            # so that loss can take this as an argument.
            # (by slicing off the first column of the output and target tensors
            # as mentioned above)
            outputDim: int = output.shape[-1]

            output_Reshaped: Tensor = output.contiguous().view(-1, outputDim)
            trgSentence_Reshaped: Tensor = trgSentence[:, 1:].contiguous().view(-1)
            ## trgSentence shape now: (batchSize * trgSentLen - 1)
            ## output shape now: (batchSize * trgSentLen - 1, outputDim)

            # 5. Calculate gradients
            loss = lossFunction(input=output_Reshaped, target= trgSentence_Reshaped)
            #loss.backward()



            # 8. Sum the loss value to a running total
            lossPerEpoch += loss.item()

    return lossPerEpoch / len(iterator) # average loss





# %% codecell
# Time the epoch!

def clock(startTime, endTime):
    elapsedTime = endTime - startTime
    elapsedMins = int(elapsedTime / 60)
    elapsedSecs = int(elapsedTime - (elapsedMins * 60))
    return elapsedMins, elapsedSecs
# %% markdown
# ### Step 8: Train the Model

# %% codecell
# %%time

trainStartTime = time.time()

NUM_EPOCHS = 10
CLIP = 1
SAVE_DIR = '.'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'transformer_bestModel.pt')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')


bestValidLoss = float('inf')

for epoch in range(NUM_EPOCHS):

    startTime = time.time()

    trainingLoss = train(model=transformerModel,
                         iterator=trainIterator,
                         noamOpt=modifAdamOptimizer,
                         lossFunction=crossEntropyLossFunction,
                         clip=CLIP)

    validationLoss = evaluate(model=transformerModel,
                              iterator=validationIterator,
                              lossFunction=crossEntropyLossFunction)

    endTime = time.time()

    epochMins, epochSecs = clock(startTime, endTime)

    if validationLoss < bestValidLoss:
        bestValidLoss = validationLoss
        torch.save(transformerModel.state_dict(), MODEL_SAVE_PATH)


    print(f'Epoch: {epoch+1:03} | Time: {epochMins}m {epochSecs}s')
    print(f'\tTrain Loss: {trainingLoss:.3f} | Train PPL: {math.exp(trainingLoss):7.3f}')
    print(f'\t Val. Loss: {validationLoss:.3f} |  Val. PPL: {math.exp(validationLoss):7.3f}')



trainEndTime = time.time()
totalMins, totalSecs = clock(trainStartTime, trainEndTime)

print("Total training time = {} mins {} secs".format(totalMins, totalSecs))

# TODO: when you train the model next time make sure to have the epoch results kept! Must export this .py file to ipynb somehow


# %% markdown
# We'll load the parameters (state_dict) that gave our model the best
# validation loss and run it the model on the test set.

# %% codecell
transformerModel.load_state_dict(torch.load(MODEL_SAVE_PATH))

testLoss = evaluate(model = transformerModel, iterator = testIterator,
                    lossFunction = crossEntropyLossFunction)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
