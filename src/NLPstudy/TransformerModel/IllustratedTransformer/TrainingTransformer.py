
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
t = torch.tensor([1,2,3])
t.double()

#torch.exp(t)
torch.exp(t.double())
torch.exp(t.float())

result = torch.exp(torch.FloatTensor(t.numpy()))
result

t.float() * result

torch.sin(t.float() * result)



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
# ### Step 2: Count Parameters amd Initialize Weights
#
# We can also see that the model has almost twice as many parameters as the attention based model (20m to 37m).
# %% codecell
def countParameters(model: Transformer):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initWeights(model:Transformer):
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

print(f'The model has {countParameters(transformerModel):,} trainable parameters')
# %% markdown
# ### Step 3: Define Optimizer
# %% codecell
adamOptimizer = optim.Adam(seqCNNModel.parameters())
adamOptimizer
# %% markdown
# ### Step 4: Define the Loss Function (Cross Entropy)
# Make sure to ignore the loss on <pad> tokens.
# %% codecell
TRG_PAD_IDX
# %% codecell
crossEntropyLossFunction = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
crossEntropyLossFunction
# %% markdown
# ### Step 5: Define Training Loop
#
# Then, we define the training loop for the model.
#
# We handle the sequences a little differently than previous tutorials. For all models we never put the `<eos>` into the `Decoder`. This is handled in the RNN models by the having the `Decoder` loop not reach having the `<eos>` as an input to the `Decoder`. But in this CNN model, we simply slice the `<eos>` token off the end of the sequence. Thus:
#
# $$\begin{align*}
# \text{trg} &= [sos, x_1, x_2, x_3, eos] \\
#     \text{trg[:-1]} &= [sos, x_1, x_2, x_3]
# \end{align*}$$
# where $x_i$ = the actual target sequence element.
#
# We then feed this into the model to get a predicted sequence that should hopefully predict the `<eos>` token:
#
# $$\begin{align*}
# \text{output} &= [y_1, y_2, y_3, eos]
# \end{align*}$$
# where $y_i$ = the predicted target sequence element. We then calculate our loss using the original `trg` tensor with the `<sos>` token sliced off the front, leaving the `<eos>` token:
#
# $$\begin{align*}
# \text{output} &= [y_1, y_2, y_3, eos]\ \
#     \text{trg[1:]} &= [x_1, x_2, x_3, eos]
# \end{align*}$$
#
# We then calculate our losses and update our parameters as is standard.
# %% codecell
def train(seqModel: Seq2Seq, iterator, optimizer, lossFunction, clip: int):

    seqModel.train() # put model in training mode

    lossPerEpoch: int = 0

    for epoch, batch in enumerate(iterator):

        # 1. Getting source and target sentences from batch
        srcSentence: Tensor = batch.src
        trgSentence: Tensor = batch.trg

        # 2. Zero the gradients from the last batch
        optimizer.zero_grad()

        # 3. Feed the source and target sentences into the seq2seq model
        # to get the output tensor of predictions.
        output, _ = seqModel(src = srcSentence, trg = trgSentence[:, :-1])
        ### trgSentence = tensor of shape (batchSize, trgSentenceLen)
        ### output = tensor of shape (batchSize, trgSentenceLen - 1, outputDim)

        # 4. Need to flatten the outputs (by slicing off the first column
        # of the output and target tensors
        # as mentioned above)
        outputDim: int = output.shape[-1]

        output: Tensor = output.contiguous().view(-1, outputDim)
            #output[1:].view(-1, outputDim)
        trgSentence: Tensor = trgSentence[:, 1:].contiguous().view(-1)
            #trgSentence[1:].view(-1)
        ## output shape now: (batchSize * trgLen - 1, outputDim)
        ## trgSentence shape now: (batchSize * trgLen - 1)

        # 5. Calculate gradients
        loss = lossFunction(input=output, target= trgSentence)
        loss.backward()

        # 6. Clip gradient so it doesn't explode
        torch.nn.utils.clip_grad_norm_(parameters = seqModel.parameters(),
                                       max_norm = clip)

        # 7. Update parameters of model
        optimizer.step()

        # 8. Sum the loss value to a running total
        lossPerEpoch += loss.item()

    return lossPerEpoch / len(iterator) # average loss
# %% markdown
# ### Step 7: Define Evaluation Loop
#
# The evaluation loop is the same as the training loop, just without the gradient calculations and parameter updates.
# %% codecell
def evaluate(seqModel: Seq2Seq, iterator, lossFunction):

    seqModel.eval()

    lossPerEpoch = 0

    with torch.no_grad():

        for epoch, batch in enumerate(iterator):

            srcSentence: Tensor = batch.src
            trgSentence: Tensor = batch.trg

            # Turn off teacher forcing
            output, _ = seqModel(src=srcSentence, trg=trgSentence[:, :-1])
            ### trgSentence = tensor of shape (batchSize, trgSentenceLen)
            ### output = tensor of shape (batchSize, trgSentenceLen - 1, outputDim)

            # 4. Need to flatten the outputs (by slicing off the first column
            # of the output and target tensors
            # as mentioned above)
            outputDim: int = output.shape[-1]

            output: Tensor = output.contiguous().view(-1, outputDim)
            #output[1:].view(-1, outputDim)
            trgSentence: Tensor = trgSentence[:, 1:].contiguous().view(-1)
            #trgSentence[1:].view(-1)
            ## output shape now: (batchSize * trgLen - 1, outputDim)
            ## trgSentence shape now: (batchSize * trgLen - 1)

            # 5. Calculate loss but not gradients since this is evaluation mode
            loss = lossFunction(input=output, target= trgSentence)


            # 6. Sum the loss value to a running total
            lossPerEpoch += loss.item()

        return lossPerEpoch / len(iterator) # average loss
# %% codecell
# Time the epoch!

def epochTimer(startTime, endTime):
    elapsedTime = endTime - startTime
    elapsedMins = int(elapsedTime / 60)
    elapsedSecs = int(elapsedTime - (elapsedMins * 60))
    return elapsedMins, elapsedSecs
# %% markdown
# ### Step 8: Train the Model
#
# Finally, we train our model.
#
# Although we have almost twice as many parameters as the attention based RNN model, it actually takes around half the time as the standard version and about the same time as the packed padded sequences version. This is due to all calculations being done in parallel using the convolutional filters instead of sequentially using RNNs.
#
# **Note**: this model always has a teacher forcing ratio of 1, i.e. it will always use the ground truth next token from the target sequence. This means we cannot compare perplexity values against the previous models when they are using a teacher forcing ratio that is not 1. See [here](https://github.com/bentrevett/pytorch-seq2seq/issues/39#issuecomment-529408483) for the results of the attention based RNN using a teacher forcing ratio of 1.
# %% codecell
%%time

NUM_EPOCHS = 10
CLIP = 1

bestValidLoss = float('inf')

for epoch in range(NUM_EPOCHS):

    startTime = time.time()

    trainingLoss = train(seqModel=seqCNNModel,
                         iterator=trainIterator,
                         optimizer=adamOptimizer,
                         lossFunction=crossEntropyLossFunction,
                         clip=CLIP)

    validationLoss = evaluate(seqModel=seqCNNModel,
                              iterator=validationIterator,
                              lossFunction=crossEntropyLossFunction)

    endTime = time.time()

    epochMins, epochSecs = epochTimer(startTime , endTime)

    if validationLoss < bestValidLoss:
        bestValidLoss = validationLoss
        torch.save(seqCNNModel.state_dict(), 'tut5_bestModel.pt')


    print(f'Epoch: {epoch+1:02} | Time: {epochMins}m {epochSecs}s')
    print(f'\tTrain Loss: {trainingLoss:.3f} | Train PPL: {math.exp(trainingLoss):7.3f}')
    print(f'\t Val. Loss: {validationLoss:.3f} |  Val. PPL: {math.exp(validationLoss):7.3f}')
# %% codecell
# We'll load the parameters (state_dict) that gave our model the best
# validation loss and run it the model on the test set.

seqCNNModel.load_state_dict(torch.load('tut4_bestModel.pt'))

testLoss = evaluate(seqModel=seqCNNModel,
                    iterator= testIterator,
                    lossFunction= crossEntropyLossFunction)

# show test loss and calculate test perplexity score:
print(f'| Test Loss: {testLoss:.3f} | Test PPL: {math.exp(testLoss):7.3f} |')
