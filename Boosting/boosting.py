from decision_stump import decisionStump
from csv_reader import load
import math
from utils import normalize, draw, sign


def boosting(samples, weaklearner, rounds):
	dist = normalize([1.] * len(samples))
	hypotheses = [None] * rounds
	alpha = [0] * rounds

	for t in range(rounds):
		def drawSample():
			return samples[draw(dist)]

		hypotheses[t] = weaklearner(drawSample)
		hypothesisResult, error = calculateError(hypotheses[t], samples, dist)
		alpha[t] = 0.5 * math.log((1-error)/(.0001 + error))
		dist = normalize([d * math.exp(-h*alpha[t]) for (d,h) in zip(dist, hypothesisResult)])
		print("Round %d, error %0.3f , alpha %0.3f" % (t, error, alpha[t]))

	def finalHypothesis(x):
		return sign(sum(a * h(x) for (a,h) in zip(alpha, hypotheses)))

	return finalHypothesis


def calculateError(h, samples, weights=None):
	if weights == None:
		weights = [1.] * len(samples)

	hypothesisResult = [ y*h(x) for (x,y) in samples]
	error = sum(w for (z,w) in zip(hypothesisResult, weights) if z==-1)
	return hypothesisResult, error

train = load()
rounds = int(raw_input('Enter the no. of rounds: '))
weaklearner = decisionStump
h = boosting(train, weaklearner, rounds)  # h is the hypothesis
print(h([2,90,60,0,0,23.5,0.191,25]))     # predict the label for this point and print -1 or +1
