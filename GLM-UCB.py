from math import exp, log, sqrt
import numpy as np
import scipy.optimize


class GLMUCB(object):
    """
    Default link function is logistic (exp(x) / (1+exp(x)))
    """
    def __init__(self, arm_list, dim=6, link_func=lambda x: exp(x)/(1+exp(x))):
        self.arm_list = arm_list
        self.t = 1
        self.dim = dim
        self.d = dim**2
        self.link = link_func
        self.M = np.identity(self.d)
        self.rewards = []
        self.context = []
        self.pulls = {arm: 0 for arm in arm_list}
        self.prev = np.zeros(self.d)

    def get_arm(self, arms, context, features):
        for arm in arms:  # initialize
            if self.pulls[arm] == 0:
                return arm

        theta_hat = scipy.optimize.root(self.to_optimize,
                                        self.prev).x
        est_rew = [(a, self.calc_reward(theta_hat, features[a]))
                   for a in arms]

        self.prev = theta_hat
        return max(est_rew, key=lambda x: x[1])[0]

    def pull_arm(self, arm, feedback, context, features):
        feat = features[arm].flatten()
        self.rewards.append(feedback)
        self.context.append(feat)
        self.M += np.outer(feat, feat)
        self.pulls[arm] += 1
        self.t += 1

    def calc_reward(self, theta, context):
        # should improve the slowly increasing function
        ma = np.array(context).flatten()
        t = len(self.rewards)

        exploration_bonus = (sqrt(2 * log(t)) *  # as per section 4.1
                         sqrt(np.dot(np.dot(ma.T, np.linalg.inv(self.M)),
                                     ma)))

        return self.link(np.inner(ma, theta)) + explore_bonus

    def to_optimize(self, theta):
        to_sum = []
        for t in range(len(self.rewards)):
            to_sum.append(self.rewards[t] -
                          self.link(np.inner(self.context[t], theta)) *
                          self.context[t])

        return np.sum(to_sum, 0)