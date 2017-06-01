import numpy as np
from csv_reader import load

class Stump(object):
   def __init__(self):
      self.posLabel = None
      self.negLabel = None
      self.threshold = None
      self.feature_index = None

   def classify(self, point):
      if point[self.feature_index] >= self.threshold:
         return self.posLabel
      else:
         return self.negLabel

   def __call__(self, point):
      return self.classify(point)


# returns the label that appears most no. of times
def majorityVote(data):
 
   labels = [label for (point, label) in data]
   try:
      return max(set(labels), key=labels.count)
   except:
      return -1


# computes the error of hypotheses
def errorOfHypothes(data, h):
   posData, negData = ([(x, y) for (x, y) in data if h(x) == 1],
                       [(x, y) for (x, y) in data if h(x) == -1])

   posError = np.sum(y == -1 for (x, y) in posData) + np.sum(y == 1 for (x, y) in negData)
   negError = np.sum(y == 1 for (x, y) in posData) + np.sum(y == -1 for (x, y) in negData)
   return min(posError, negError) / len(data)


# Computes the best threshold for a given feature. Returns (threshold, error)
def bestThreshold(data, index, errorFunction):
   

   thresholds = [point[index] for (point, label) in data]
   def makeThreshold(t):
      return lambda x: 1 if x[index] >= t else -1

   errors = [(threshold, errorFunction(data, makeThreshold(threshold))) for threshold in thresholds]
   return min(errors, key=lambda p: p[1])


def defaultError(data, h):
   return errorOfHypothes(data, h)


# finds the index of the best feature to split on, and the best threshold for
# that index
def decisionStump(drawExample, errorFunction=defaultError):

   data = [drawExample() for _ in range(500)]
   data = np.array(data)
   bestThresholds = [(i,) + bestThreshold(data, i, errorFunction) for i in range(len(data[0][0]))]
   feature_index, threshold, _ = min(bestThresholds, key = lambda p: p[2])

   stump = Stump()
   stump.feature_index = feature_index
   stump.threshold = threshold
   stump.posLabel = majorityVote([x for x in data if x[0][feature_index] >= threshold])
   stump.negLabel = majorityVote([x for x in data if x[0][feature_index] < threshold])

   return stump
