"""
CODE TAKEN FROM: https://github.com/pytorch/text/blob/master/torchtext/data/metrics.py
"""

import math
import collections
import collections.Counter as Counter
import torch
import torch.tensor as Tensor
from torchtext.data.utils import ngrams_iterator



def computeNGramCounter(tokens: list, MAX_N: int) -> Counter:
    """ Create a Counter with a count of unique n-grams in the tokens list
    Arguments:
        tokens: a list of tokens (typically a string split on whitespaces)
        MAX_N: the maximum order of n-gram wanted
    Outputs:
        output: a collections.Counter object with the unique n-grams and their
            associated count
    Examples:
        >> from torchtext.data.metrics import _compute_ngram_counter
        >> tokens = ['me', 'me', 'you']
        >> _compute_ngram_counter(tokens, 2)
            Counter({('me',): 2,
             ('you',): 1,
             ('me', 'me'): 1,
             ('me', 'you'): 1,
             ('me', 'me', 'you'): 1})
    """
    assert MAX_N > 0

    ngramsCounter: Counter = Counter(tuple(x.split(' '))
                                         for x in ngrams_iterator(tokens, MAX_N))

    return ngramsCounter






def bleuScore(candidateCorpus,
              referencesCorpus,
              MAX_N: int = 4, weights: list = [0.25] * 4) -> float:
    """Computes the BLEU score between a candidate translation corpus and a references
    translation corpus. Based on https://www.aclweb.org/anthology/P02-1040.pdf
    Arguments:
        candidateCorpus: an iterable of candidate translations. Each translation is an
            iterable of tokens
        referencesCorpus: an iterable of iterables of reference translations. Each
            translation is an iterable of tokens
        MAX_N: the maximum n-gram we want to use. E.g. if max_n=3, we will use unigrams,
            bigrams and trigrams
        weights: a list of weights used for each n-gram category (uniform by default)
    Examples:
        >> from torchtext.data.metrics import bleu_score
        >> candidate_corpus = [['I', 'ate', 'the', 'apple'], ['I', 'did']]
        >> references_corpus = [[['I', 'ate', 'it'], ['I', 'ate', 'apples']],
                [['I', 'did']]]
        >> bleu_score(candidate_corpus, references_corpus)
            0.7598356856515925
    """

    # Assertion 1:
    assert MAX_N == len(weights), 'Length of the "weights" list has be equal to max_n'

    # Assertion 2:
    assert len(candidateCorpus) == len(referencesCorpus), \
        'The length of candidate and reference corpus should be the same'

    clippedCounts: Tensor = torch.zeros(MAX_N)
    totalCounts: Tensor = torch.zeros(MAX_N)
    weights: Tensor = torch.tensor(weights)

    candidateLen: int = 0.0
    refsLen: int = 0.0

    for (candidate, refs) in zip(candidateCorpus, referencesCorpus):
        candidateLen += len(candidate)

        # Get the length of the reference that's closest in length to the candidate
        refsLenList: list = [float(len(ref)) for ref in refs]
        refsLen += min(refsLenList, key=lambda x: abs(len(candidate) - x))


        referenceCounters: Counter = computeNGramCounter(refs[0], MAX_N)

        for ref in refs[1:]:
            referenceCounters = referenceCounters | computeNGramCounter(ref, MAX_N)

        candidateCounter: Counter = computeNGramCounter(candidate, MAX_N)

        clippedCounter: Counter = candidateCounter & referenceCounters

        for ngram in clippedCounter:
            iLast: int = len(ngram) - 1
            clippedCounts[iLast] += clippedCounter[ngram]

        for ngram in candidateCounter:  # TODO: no need to loop through the whole counter
            iLast: int = len(ngram) - 1
            totalCounts[iLast] += candidateCounter[ngram]

    if min(clippedCounts) == 0:
        return 0.0
    else:
        pn = clippedCounts / totalCounts
        logPN = weights * torch.log(pn)
        score = torch.exp(sum(logPN))

        bp = math.exp(min(1 - refsLen / candidateLen, 0))

        return bp * score.item()
