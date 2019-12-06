# %% markdown
#
# %% markdown
# Deep learning consists of composing linearities with non-linearities in clever ways. The introduction of non-linearities allows for powerful models. In this section, we will play with these core components, make up an objective function, and see how the model is trained.
# %% codecell
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
# %% codecell
lin = nn.Linear(5, 3) # maps from R^5 to R^3
lin
# %% codecell
data = autograd.Variable(torch.randn(2, 5))
print(lin(data))
# %% codecell
# In pytorch, most non-linearities are in torch.functional (we have it imported as F)
# Note that non-linearites typically don't have parameters like affine maps do.
# That is, they don't have weights that are updated during training.
data = autograd.Variable(torch.randn(2, 2))
print(data)
print(F.relu(data))
# %% codecell
# Softmax is also in torch.nn.functional
data = autograd.Variable(torch.randn(5))
print(data)
print(F.softmax(data))
print(F.softmax(data).sum())  # Sums to 1 because it is a distribution!
print(F.log_softmax(data))  # theres also log_softmax
# %% codecell
# Example: Logistic Regression bag of Words Classifier.

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

testData = [("Yo creo que si".split(), "SPANISH"),
            ("it is lost on me".split(), "ENGLISH")]

# mapping each word in the vocab to a unique integer which will be
# its index into the bag of words vector
wordToIx = {}

for sent, _ in data + testData:
    for word in sent:
        if word not in wordToIx:
            wordToIx[word] = len(wordToIx)

print(wordToIx)

VOCAB_SIZE = len(wordToIx)
NUM_LABELS = 2

# %% codecell
class BoWClassifier(nn.Module): # inheriting from nn.Module

    def __init__(self, numLabels, vocabSize):
        # Calls the init function of nn.Module
        super(BoWClassifier, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # nn.Linear() provides the affine map.
        self.linear = nn.Linear(vocabSize, numLabels)

    def forward(self, bowVector):
        # Pass the input through the linear layer, then pass
        # that through log_softmax
        return F.log_softmax(self.linear(bowVector))



def makeBowVector(sentence, wordToIx):
    vec = torch.zeros(len(wordToIx))

    for word in sentence:
        vec[wordToIx[word]] += 1

    return vec.view(1, -1)

def makeTarget(label, labelToIx):
    return torch.LongTensor([labelToIx[label]])
# %% codecell
model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)


# BoWClassifier (our module) will store knowledge of the nn.Linear's
# parameters (? when you assign a component to a class variable in the
# __init__ function of our module?)
for param in model.parameters():
    print(param) # parameters are: numlabels, and vocabsize

    # first output below is A (matrix)
    # the second output is b (vector)
    # where the affine linear map is f(x) = Ax + b
# %% codecell
# To run the model, pass in a BoW vector but wrapped in an
# autograd.Variable
sample = data[0]
print("sample: ", sample)
print("sample[0]: ", sample[0])

bowVector = makeBowVector(sample[0], wordToIx)
print("bowVector: ", bowVector)

# TODO: does calling model call forward pass as well?
# How else are the log softmax probs below obtained?
logProbabilities = model(autograd.Variable(bowVector))
print("logprobs: ", logProbabilities)
# %% codecell
# Defining which of the above values corresponds to the log
# probability of ENGLISH and which to SPANISH
labelToIx = {"SPANISH": 0, "ENGLISH": 1}

print(labelToIx)
print(wordToIx)
# %% codecell
 #Testing the model first:
# 1. pass instances through to get log probabilities,
# 2. compute a  loss function
# 3. compute gradient of loss function
# 4. update parameters with a graident step

# Run on test data before we train, as a sample before-after showcase

# Step 1: get log probs
for instance, label in testData:
    bowVec = autograd.Variable(makeBowVector(instance, wordToIx))
    logProbs = model(bowVec)

    print("instance: ", instance,
          "\nlabel: ", label,
          "\nlogProbs: ", logProbs)


# Print the matrix column corresponding to " creo"
n = next(model.parameters())
i = wordToIx["creo"]
print("\nmodel params: ", n)
print("\nmodel params? corresponding to 'creo': ", n[:, i])

# %% codecell
# Going through the forloop below manually to understand how
# bow vectors are made:

print(data)
i1, l1 = data[0]
i2, l2 = data[1]
i3, l3 = data[2]
i4, l4 = data[3]

#print(wordToIx)
b1 = autograd.Variable(makeBowVector(i1, wordToIx))
b2 = autograd.Variable(makeBowVector(i2, wordToIx))
b3 = autograd.Variable(makeBowVector(i3, wordToIx))
b4 = autograd.Variable(makeBowVector(i4, wordToIx))

print("b1: ", b1, "\nb2: ", b2, "\nb3: ", b3, "\nb4: ", b4)

t1 = autograd.Variable(makeTarget(l1, labelToIx))
t2 = autograd.Variable(makeTarget(l2, labelToIx))
t3 = autograd.Variable(makeTarget(l3, labelToIx))
t4 = autograd.Variable(makeTarget(l4, labelToIx))

print("\nt1: ", t1, "\nt2: ", t2, "\nt3: ", t3, "\nt4: ", t4)


## -----------------------------------------------------------------

# Run the forward pass to get log probabilities
logProbs1 = model(b1)
logProbs2 = model(b2)
logProbs3 = model(b3)
logProbs4 = model(b4)
print("\nlogprobs1: ", logProbs1, "\nlogprobs2: ", logProbs2,
      "\nlogprobs3: ", logProbs3, "\nlogprobs4: ", logProbs4)

# Step 2: Compute the loss function
loss1 = lossFunction(logProbs1, t1)
loss2 = lossFunction(logProbs2, t2)
loss3 = lossFunction(logProbs3, t3)
loss4 = lossFunction(logProbs4, t4)
print("\nloss1: ", loss1, "\nloss2: ", loss2,
      "\nloss3: ", loss3, "\nloss4: ", loss4)

# Step 3: compute gradients with respect to loss
loss1.backward()
loss2.backward()
loss3.backward()
loss4.backward()
#print("\nb1.grad: ", b1.grad)
# which variable has the .grad property here?

# Step 4: Update parameters with a gradient step
optimizer.step()
# %% codecell

# Preparing to train the data now:

# Steps to train the model:
# 1. pass instances through to get log probabilities,
# 2. compute a  loss function
# 3. compute gradient of loss function
# 4. update parameters with a gradient step


lossFunction = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# Usually we use 5 and 30 epochs for training data but here
# we use 100 since this data is small

NUM_ITER = 100

for epoch in range(NUM_ITER):
    for instance, label in data:
        # Pytorch accumulates gradients so need to clear
        # them out each instance
        model.zero_grad()

        # Step 1: Make the BOW vector and wrap the target in a
        # Variable as an integer. For example, if the target is
        # SPANISH, then wrap the integer 0. If target is ENGLISH
        # wrap the integer 1. (as per labelToIx)
        # The loss function then knows that the 0th element of the
        # log probabilities is the log probability corresponding
        # to SPANISH
        bowVec = autograd.Variable(makeBowVector(instance, wordToIx))
        target = autograd.Variable(makeTarget(label, labelToIx))

        # Run the forward pass to get log probabilities
        logProbs = model(bowVec)

        # Step 2: Compute the loss function
        loss = lossFunction(logProbs, target)

        # Step 3: compute gradients with respect to loss
        loss.backward()
        # Step 4: Update parameters with a gradient step
        optimizer.step()


# TODO: why getting error here?
#for instance, label in testData:
#    bowVec = autograd.Variable(instance, labelToIx)
#    logProbs = model(bowVec) # forward pass
#    print("test log probs: ", logProbabilities)


# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell

# %% codecell
