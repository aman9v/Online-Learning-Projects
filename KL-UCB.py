import numpy as np
import random
import math
from matplotlib import pyplot as plt
from scipy import optimize


class KL_UCB(object):

    """
    A bandit instance consists of k mean rewards in [0, 1] and each arm generates bernoulli 
    reward as per their mean. Generate 1000 such instances and Average the sample complexities over 
    all the instances to obtain the average sample complexity and the mistake probability should be 
    calculated as the fraction of times non best arm is returned after stop signals are received.

    """

    def __init__(self, epsilon, num_arms, alpha=2):
        self.t = 1
        self.K = num_arms
        self.N = [0] * num_arms
        self.delta = 0.1
        self.epsilon = epsilon
        self.alpha = alpha
        self.B = float("inf")
        self.U = [0.0] * num_arms
        self.L = [0.0] * num_arms
        self.means = [random.random() for arm in range(num_arms)]
        self.mu = [0.0] * num_arms
        self.rewards = [1 if random.random() <= mean  else 0 for mean in self.means]



    def start_game(self):
        
        """  
        sample every arm once and update the initial upper and 
        lower bound for each arm.  

        """
        beta = [0] * self.K+1

        for a in range(1, self.K+1):

            sample = self.rewards[a]
            self.N[a] += 1
            self.mu[a] = sample/self.N[a]

            beta[a] = self.calculate_beta(a, self.delta, self.K, self.alpha)

            self.L[a] = self.means[a] - math.sqrt(beta[a]/(2*self.N[a]))
            self.U[a] = self.means[a] + math.sqrt(beta[a]/(2*self.N[a]))


            self.t += 1

        print(beta)
        print(self.mu)    
        
        while True:
            done = False

            if self.B <= self.epsilon:
                done = True
                break
            
            # for a in range(1, self.K+1):
            #     u = self.calculate_upper(beta[a], a)           
            #     u = max([(self.objective(self.mu[a], x, beta[a]), a) for x in numpy.linspace(self.mu[a], 1, self.K)])
            #     l = self.calculate_lower(beta[a], a)
                
            
            u = max([(self.objective(self.mu[a], x, beta[a]), a) for a in range(1, self.K+1) for x in numpy.linspace(self.mu[a], 1, self.K)]    )
            l = min([(self.objective(self.mu[a], x, beta[a]), a) for a in range(1, self.K+1) for x in numpy.linspace(0, self.mu[a], self.K)]    )


            sample_u = self.rewards[u[1]]
            sample_l = self.rewards[l[1]]
            
            self.N[u[1]] += 1
            self.N[l[1]] += 1

            self.mu[u[1]] += sample_u
            self.mu[l[1]] += sample_l

            self.U[u[1]] = self.mu[u[1]] + math.sqrt(self.beta[u[1]]/(2*self.N[u[1]]))
            self.L[l[1]] = self.mu[l[1]] - math.sqrt(self.beta[l[1]]/(2*self.N[l[1]]))

            self.B = U[u[1]] - L[l[1]]
 

            self.t += 1

        print('sample complexity is %d' % self.t)
            
                                                

    # calculate KL divergence    
    def KL_div(self, p, q):
        v = p * math.log(p/q) + (1-p)*math.log((1-p)/(1-q))
        return v
    
    def objective(self, pa, q, beta):

        return pa * math.log(pa/q) + (1-pa)*math.log((1-pa)/(1-q)) - (beta/self.N[a])
    
    def calculate_upper(self, beta, a):
        # bnds = ((self.mu[a], 1))        
        # solution_p = optimize.minimize(lambda q: (beta/self.N[a])-self.KL_div(self.mu[a],q), 1, method='SLSQP',)
        # return solution_p.q
        return optimize.fsolve(self.objective, 0, args=(self.mu[a], beta))

    def calculate_lower(self, beta, a):
        # solution_q = optimize.minimize(lambda q: (beta/self.N[a])-self.KL_div(self.mu[a],q), 1, method='SLSQP',)
        # return solution_q.q
        return optimize.fsolve(self.objective, 0, args=(self.mu[a], beta))
        

    def calculate_beta(self, t, delta, K, alpha, K1=4*math.e+4):
        v = math.log((K1*K*(t**alpha))/delta) + math.log(math.log(K1*K*(t**alpha)/delta))
        return v        


bandit_instance = KL_UCB(0.01, 10)
bandit_instance.start_game()




class Lil_UCB(object):
    

    def __init__(self, epsilon, delta, lamda, beta, sigma, num_arms):
        self.arms = range(num_arms)
        # self.T = [0] * num_arms
        # self.t = 0
        self.epsilon = epsilon
        self.delta = delta
        self.sigma = sigma
        self.lamda = lamda
        self.beta = beta
        self.mu = [0.0] * num_arms
        self.means = [random.random() for arm in range(num_arms)]
        # self.means = [1/2 if arm == 0 else ((1/2)-(arm/70)) for arm in range(num_arms)]
        # self.rewards = [1 if random.random() <= mean  else 0 for mean in self.means]



    def initialize(self):
        self.T = [0] * len(self.arms)
        self.t = 0
        self.rewards = [1 if random.random() <= mean  else 0 for mean in self.means]        
        for arm in self.arms:
            self.T[arm] = 1
            self.t += 1
            self.mu[arm] = float(sum([self.rewards[arm] for _ in range(self.T[arm])])/self.T[arm])
        # print(self.means, self.t, self.T)    


    def start_game(self):
        delta_arm = [max(self.means)-self.means[arm] for arm in self.arms]
        while True:
            done = False
            
            total_pulls = sum(self.T)
            for arm in self.arms:
                if self.T[arm] >= 1+self.lamda*(total_pulls-self.T[arm]):
                    done = True
                    break

            if done:
                break

            index = 0  # holds the index of the best arm. 
            upper_bound = 0

            for arm in self.arms:
                temp = math.sqrt(2*(self.sigma**2)* (1+self.epsilon) * math.log((math.log((1+self.epsilon)* self.T[arm]+2))/self.delta))
                temp = self.means[arm] + (1+self.beta)*(1+math.sqrt(self.epsilon))*temp
                # print(temp)
                if temp > upper_bound:
                     
                    upper_bound = temp
                    index = arm

            self.T[index] += 1
            # print(self.T)
            # self.means[index] = float(sum([self.rewards[index] for _ in range(self.T[index])])/self.T[index])
            self.mu[index] = float(sum([self.rewards[index] for _ in range(self.T[index])])/self.T[index])
            # self.mu[index] = ((self.T[index]-1)*self.mu[index] + self.rewards[index]) / self.T[index] #average the rewards
            # print(index)
            # print(self.means)
            self.t += 1
        
        
        empercial_best = max(self.mu)
        best_arm_index = [i for i,j in enumerate(self.mu) if j == empercial_best]
        # print(best_arm_index==index)
        best_arm = self.arms[best_arm_index[0]]

        # print("For %d arms -> BEST_ARM: %d\tMean: %f\tIterations: %d \t " % (len(self.arms), best_arm, self.means[best_arm], self.t))
        # log_file.write("ITERATION: %6d\tBEST_ARM: %s\tCONFIG_MU: %f\tDELTA: %f\n" %(timestep, str(best_arm), best_arm.get_config_mu(), best_arm.get_delta()))
        return self.t, best_arm



sample_comp = []
for k in range(10, 51, 10):
    sample_complexity = []
    bandit_instance = Lil_UCB(0.01, 0.1, 9, 1, 0.5, k)
    
    for _ in range(1000):
        bandit_instance.initialize()
        t, best_arm = bandit_instance.start_game()
        sample_complexity.append(t)
        
    average = sum(sample_complexity)/float(len(sample_complexity))
    sample_comp.append(average)
    print("Sample Complexity for %d arms\t %d" % (k, average))


ax1 = plt.subplot(211)
plt.ylabel('Mean Sample Complexity')
plt.xlabel('Arms(K)')     
plt.plot(range(10, 51, 10), sample_comp)
plt.plot([10,20,30,40,50], sample_comp, 'ro')
plt.show()

plt.show()    
# bandit_instance = Lil_UCB(0.01, 0.1, 9, 1, 0.5, 10)
# bandit_instance.initialize()
# bandit_instance.start_game()


