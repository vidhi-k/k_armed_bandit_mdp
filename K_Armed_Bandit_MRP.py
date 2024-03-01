import numpy as np
import matplotlib.pyplot as plt

class Bandit(object):
    def __init__(self, trueActionValues, epsilon , alpha, c, kArms=10):
        self.numberOfArms = kArms
        self.trueActionValues = trueActionValues
        self.epsilon = epsilon
        self.stepsToBeTaken = 1000
        self.totalRewards = []
        self.meanActionValuesForEachArm = np.zeros(kArms)
        self.numberOfTimesEachActionTaken = np.zeros(kArms)
        self.alphaForOptimistic = alpha
        self.c = c

    #action selected on the basis of epsilon action selection policy
    def select_action(self):
        currentProb = np.random.rand()
        if currentProb < self.epsilon:
            return np.random.choice(self.numberOfArms)
        else:
            return np.argmax(self.meanActionValuesForEachArm)

    #random reward award with mean as trueActionMean
    def select_reward(self, action):
        return np.random.normal(self.trueActionValues[action], 1)
    
    def update_mean_action_value(self, action, reward):
        self.numberOfTimesEachActionTaken[action] += 1
        self.meanActionValuesForEachArm[action] += (reward - self.meanActionValuesForEachArm[action]) / self.numberOfTimesEachActionTaken[action]        

    def update_mean_action_values_for_optimistic(self, action, reward):
        self.numberOfTimesEachActionTaken[action] += 1
        self.meanActionValuesForEachArm[action] = self.meanActionValuesForEachArm[action] + self.alphaForOptimistic * (reward - self.meanActionValuesForEachArm[action])

    #simulation of one run for epsilon greedy action selection
    def start_game(self, optimalAction):
        for step in range(self.stepsToBeTaken):
            action = self.select_action()
            reward = self.select_reward(action)
            self.update_mean_action_value(action, reward)
            self.totalRewards.append(reward)
            optimalAction[step] += int(action == np.argmax(self.trueActionValues))

    #simulation for one run of optimistic initial value, where initial Q value is taken as 5
    def optimistic_value(self, optimalAction):
        self.meanActionValuesForEachArm = np.full(10, 5.0)
        for step in range(self.stepsToBeTaken):
            action = np.argmax(self.meanActionValuesForEachArm)
            reward = self.select_reward(action)
            self.update_mean_action_values_for_optimistic(action, reward)
            self.totalRewards.append(reward)
            optimalAction[step] += int(action == np.argmax(self.trueActionValues))

    #simulation for one run of ucb
    def ucb_start(self, optimalAction):
        for step in range(self.stepsToBeTaken):
            ucb_values = self.meanActionValuesForEachArm + self.c * np.sqrt(np.log(step + 1) / (self.numberOfTimesEachActionTaken + 1e-5))
            action = np.argmax(ucb_values)
            reward = self.select_reward(action)
            self.update_mean_action_value(action, reward)
            self.totalRewards.append(reward)
            optimalAction[step] += int(action == np.argmax(self.trueActionValues))
            
class MRP:
    def __init__(self, alpha, eps):
        self.states = ['F1', 'A', 'B', 'C', 'D', 'E', 'F1']
        self.trueActionValues = [0/6, 1/6, 2/6, 3/6, 4/6, 5/6, 6/6]
        self.alpha = alpha
        self.noOfActionToBeTaken = eps
        self.noOfTimesAStateIsVisited = np.full(7, 0)
        self.errs_rms = []

    def update_values(self, state, nextState, reward, actionValue):
        actionValue[state] += self.alpha * (reward + actionValue[nextState] - actionValue[state])

    #calculates rms errors
    def cal_rms(self, actionValue, trueActionValues):
        squareError = (actionValue - trueActionValues) ** 2
        meanSquare = np.mean(squareError)
        rmse = np.sqrt(meanSquare)
        return rmse

    #simulations for alphas other than 1/n
    def start_game(self):
        actionValues = np.full(len(self.states), 0.5)
        actionValues[0] = 0
        actionValues[6] = 0

        for i in range(self.noOfActionToBeTaken):
            s = 3
            while True:
                self.noOfTimesAStateIsVisited[s] += 1
                action = np.random.choice([-1,1])
                nextstate = s + action
                if nextstate == 6:
                    reward = 1
                else: 
                    reward = 0
                
                self.update_values(s, nextstate, reward, actionValues)

                if nextstate <= 0 or nextstate >= 6:
                    break
                s = nextstate

            rmse = self.cal_rms(actionValues[1:6], np.array(self.trueActionValues[1:6]))
            self.errs_rms.append(rmse)
    
    #simlulation for alpha = 1/n
    def start_game_for_sample_avg(self):
        actionValues = np.full(len(self.states), 0.5)
        actionValues[0] = 0
        actionValues[6] = 0

        for i in range(self.noOfActionToBeTaken):
            s = 3
            while True:
                self.noOfTimesAStateIsVisited[s] += 1
                action = np.random.choice([-1,1])
                nextstate = s + action
                if nextstate == 6:
                    reward = 1
                else: 
                    reward = 0
                
                actionValues[s] += (1/self.noOfTimesAStateIsVisited[s]) * (reward + actionValues[nextstate] - actionValues[s])

                if nextstate <= 0 or nextstate >= 6:
                    break
                s = nextstate

            rmse = self.cal_rms(actionValues[1:6], np.array(self.trueActionValues[1:6]))
            self.errs_rms.append(rmse)
    

# Run this program from a terminal and see the output
# Please make sure that the program you submit can be run from a terminal
def main():

    #epsilon greedy action selection
    runs = 2000
    rewards_all_runs = []
    optimalAction = np.full(1000, 0) #to find optimal action percentage at each ith step
    print(f"Running simulations for average reward in epsilon greedy action selection")
    print(f"Running simulation with epsilon=0.1...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0.1, 0, 0)
        b1.start_game(optimalAction)
        rewards_all_runs.append(b1.totalRewards)
    
    plt.figure(figsize=(10, 5))
    rewards_mean = np.mean(rewards_all_runs, axis=0)
    plt.plot(rewards_mean, label=f"Epsilon={0.1}")

    rewards_all_runs = []
    optimalAction = np.full(1000, 0)
    print(f"Running simulation with epsilon=0.01...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0.01, 0, 0)
        b1.start_game(optimalAction)
        rewards_all_runs.append(b1.totalRewards)
    
    rewards_mean = np.mean(rewards_all_runs, axis=0)
    plt.plot(rewards_mean, label=f"Epsilon={0.01}")

    rewards_all_runs = []
    optimalAction = np.full(1000, 0)
    print(f"Running simulation with epsilon=0.0...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0.0, 0, 0)
        b1.start_game(optimalAction)
        rewards_all_runs.append(b1.totalRewards)
    
    rewards_mean = np.mean(rewards_all_runs, axis=0)
    plt.plot(rewards_mean, label=f"Epsilon={0.0}")

    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.title(f"Epsilon-Greedy 10 Armed Bandit - Average Reward")
    # plt.savefig('epsilonaveragereward.png',dpi=300)
    plt.show()
    
    plt.clf()
    plt.figure(figsize=(10, 5))
    optimalAction = np.full(1000, 0) #to find optimal action percentage at each ith step
    print(f"Running simulations for percentage optimal action in epsilon greedy action selection")
    print(f"Running simulation with epsilon=0.1...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0.1, 0, 0)
        b1.start_game(optimalAction)
    
    plt.plot((optimalAction/2000) * 100, label=f"Epsilon={0.1}")

    optimalAction = np.full(1000, 0)
    print(f"Running simulation with epsilon=0.01...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0.01, 0, 0)
        b1.start_game(optimalAction)

    plt.plot((optimalAction/2000) * 100, label=f"Epsilon={0.01}")

    optimalAction = np.full(1000, 0)
    print(f"Running simulation with epsilon=0.0...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0.0, 0, 0)
        b1.start_game(optimalAction)

    plt.plot((optimalAction/2000) * 100, label=f"Epsilon={0.0}")

    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.legend()
    plt.title(f"Epsilon-Greedy 10 Armed Bandit - % Optimal Action")
    # plt.savefig('epsilonoptimalreward.png',dpi=300)
    plt.show()

#######################################################################################################
    plt.clf()
    plt.figure(figsize=(10, 5))

    #simlulation for optimistic initial value 
    rewards_all_runs = []
    optimalAction = np.full(1000, 0) #to find optimal action percentage at each ith step
    print(f"Running simulations for average reward in optimistic intial value")
    print(f"Running simulation with alpha=0.1...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0.1, 0.1, 2)
        b1.optimistic_value(optimalAction)
        rewards_all_runs.append(b1.totalRewards)
    
    plt.figure(figsize=(10, 5))
    rewards_mean = np.mean(rewards_all_runs, axis=0)
    plt.plot(rewards_mean, label=f"Alpha={0.1}")

    rewards_all_runs = []
    print(f"Running simulation with alpha=0.4...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0, 0.4, 2)
        b1.optimistic_value(optimalAction)
        rewards_all_runs.append(b1.totalRewards)
    
    rewards_mean = np.mean(rewards_all_runs, axis=0)
    plt.plot(rewards_mean, label=f"Alpha={0.4}")

    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.title(f"Optimistic Value 10 Armed Bandit - Average Reward")
    # plt.savefig('optiAvReward.png',dpi=300)
    plt.show()
    
    plt.clf()
    plt.figure(figsize=(10, 5))
    optimalAction = np.full(1000, 0) #to find optimal action percentage at each ith step
    print(f"Running simulations for optimal action percentage in optimistic initial value")
    print(f"Running simulation with alpha=0.1...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0.1, 0.1, 2)
        b1.optimistic_value(optimalAction)

    plt.plot((optimalAction/2000) * 100, label=f"Alpha={0.1}, Q = 5")

    optimalAction = np.full(1000, 0)
    print(f"Running simulation with alpha=0.4...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0, 0.4, 2)
        b1.optimistic_value(optimalAction)
    
    plt.plot((optimalAction/2000) * 100, label=f"Alpha={0.4}, Q = 5")

    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.legend()
    plt.title(f"Optimistic Value 10 Armed Bandit - % Optimal Action")
    # plt.savefig('optimOptimal.png',dpi=300)
    plt.show()

    plt.clf()
    plt.figure(figsize=(10, 5))
    optimalAction = np.full(1000, 0) #to find optimal action percentage at each ith step
    print(f"Running simulations to plot the figure 2.2")
    print(f"Running simulation with epsilon=0.1...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0.1, 0.1, 2)
        b1.start_game(optimalAction)
    
    plt.plot((optimalAction/2000) * 100, label=f"Q = 0, epsilon = 0.1, alpha = 0.1")

    optimalAction = np.full(1000, 0)
    print(f"Running simulation with alpha=0.1...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0, 0.1, 2)
        b1.optimistic_value(optimalAction)
    
    plt.plot((optimalAction/2000) * 100, label=f"Q = 5, epsilon = 0, alpha = 0.1")

    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.legend()
    plt.title(f"Figure 2.3")
    # plt.savefig('optimOptimal2.3.png',dpi=300)
    plt.show()

#############################################################################################################
    plt.clf()
    plt.figure(figsize=(10, 5))
    #ucb initial value
    #this code snippet was used to plot figure 2.4 of textbook
    # rewards_all_runs = []
    # print(f"Running simulation with epsilon=0.1...")
    # for i in range(runs):
    #     actionValues=np.random.normal(0,1,10)
    #     b1 = Bandit(actionValues, 0.1, 0, 0)
    #     b1.start_game()
    #     rewards_all_runs.append(b1.totalRewards)
    
    # rewards_mean = np.mean(rewards_all_runs, axis=0)
    # plt.plot(rewards_mean, label=f"Epsilon={0.1}")

    rewards_all_runs = []
    optimalAction = np.full(1000, 0) #to find optimal action percentage at each ith step
    print(f"Running simulations for average reward in ucb action selection")
    print(f"Running simulation with c=0...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0.1, 0.1, 0)
        b1.ucb_start(optimalAction)
        rewards_all_runs.append(b1.totalRewards)
    
    plt.figure(figsize=(10, 5))
    rewards_mean = np.mean(rewards_all_runs, axis=0)
    plt.plot(rewards_mean, label=f"c = 0")

    rewards_all_runs = []
    print(f"Running simulation with c = 2...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0, 0.1, 2)
        b1.ucb_start(optimalAction)
        rewards_all_runs.append(b1.totalRewards)
    
    rewards_mean = np.mean(rewards_all_runs, axis=0)
    plt.plot(rewards_mean, label=f"c=2")

    # plt.xlabel("Steps")
    # plt.ylabel("Average Reward")
    # plt.legend()
    # plt.title(f"UCB 10 Armed Bandit - Average Reward")
    # plt.savefig('ucbepsilonaveragereward.png',dpi=300)
    # plt.show()

    rewards_all_runs = []
    print(f"Running simulation with c=5...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0, 0.1, 5)
        b1.ucb_start(optimalAction)
        rewards_all_runs.append(b1.totalRewards)
    
    rewards_mean = np.mean(rewards_all_runs, axis=0)
    plt.plot(rewards_mean, label=f"c=5")

    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.title(f"UCB 10 Armed Bandit - Average Reward")
    # plt.savefig('ucbaveragereward.png',dpi=300)
    plt.show()
    
    plt.clf()
    plt.figure(figsize=(10, 5))
    optimalAction = np.full(1000, 0)  #to find optimal action percentage at each ith step
    print(f"Running simulations for optimal action percentage in ucb action selection")
    print(f"Running simulation with c=0...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0.1, 0.1, 0)
        b1.ucb_start(optimalAction)
    
    plt.plot((optimalAction/2000) * 100, label=f"c = 0")

    optimalAction = np.full(1000, 0)
    print(f"Running simulation with c = 2...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0, 0.1, 2)
        b1.ucb_start(optimalAction)
    
    plt.plot((optimalAction/2000) * 100, label=f"c = 2")

    optimalAction = np.full(1000, 0)
    print(f"Running simulation with c = 5...")
    for i in range(runs):
        actionValues=np.random.normal(0,1,10)
        b1 = Bandit(actionValues, 0, 0.1, 5)
        b1.ucb_start(optimalAction)

    plt.plot((optimalAction/2000) * 100, label=f"c = 5")

    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.legend()
    plt.title(f"UCB 10 Armed Bandit - % Optimal Action")
    # plt.savefig('ucboptimal.png',dpi=300)
    plt.show()

##########################################################################################################    
    # alphas for Markov Reward Process
    print(f"Simulations for MRP")
    alpha = [0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
    plt.figure(figsize=(10, 5))
    for a in alpha:
        m = MRP(a, 100)
        m.start_game()
        plt.plot(m.errs_rms, label=f"Alpha={a}")
    
    # plt.figure(figsize=(10, 5))
    m = MRP(0.1, 100)
    m.start_game_for_sample_avg()
    plt.plot(m.errs_rms, label=f"Alpha=1/n")

    plt.xlabel("Steps")
    plt.ylabel("Average RMS error")
    plt.legend()
    plt.title(f"TD(0) algorithm for 100 steps")
    # plt.savefig('qu2_100.png',dpi=300)
    plt.show()

    plt.clf()
    plt.figure(figsize=(10, 5))
    for a in alpha:
        m = MRP(a, 500)
        m.start_game()
        plt.plot(m.errs_rms, label=f"Alpha={a}")
    
    m = MRP(0.1, 500)
    m.start_game_for_sample_avg()
    plt.plot(m.errs_rms, label=f"Alpha=1/n")

    plt.xlabel("Steps")
    plt.ylabel("Average RMS error")
    plt.legend()
    plt.title(f"TD(0) algorithm for 500 steps")
    # plt.savefig('q2_500.png',dpi=300)
    plt.show()

    plt.clf()
    plt.figure(figsize=(10, 5))
    for a in alpha:
        m = MRP(a, 1000)
        m.start_game()
        plt.plot(m.errs_rms, label=f"Alpha={a}")

    m = MRP(0.1, 1000)
    m.start_game_for_sample_avg()
    plt.plot(m.errs_rms, label=f"Alpha=1/n")

    plt.xlabel("Steps")
    plt.ylabel("Average RMS error")
    plt.legend()
    plt.title(f"TD(0) algorithm for 1000 steps")
    plt.savefig('q2_1000.png',dpi=300)
    plt.show()
 


if __name__=='__main__':
    main()