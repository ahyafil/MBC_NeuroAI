import numpy as np
from scipy import optimize
from abc import ABC, abstractmethod
import requests
import pandas as pd


def import_choice_engineer_competition_data():
    # GitHub API URL for the directory contents
    api_url = "https://api.github.com/repos/ohaddan/competition/contents/data/1_6_vs_2_6"
    
    # Fetch the directory contents
    response = requests.get(api_url)
    files = response.json()
    
    # Filter and list the first 20 CSV files
    csv_files = [file['name'] for file in files if file["name"].endswith(".csv")]
    
    # Now load all of them and merge in single df
    dataframes = []
    
    # Loop through each CSV file, read it into a DataFrame, and append to the list
    datadir = 'https://raw.githubusercontent.com/ohaddan/competition/master/data/1_6_vs_2_6/'
    pd.set_option('future.no_silent_downcasting', True)
    for (n,file) in enumerate(csv_files):
        df = pd.read_csv(datadir+file)  # Read the CSV file
        df['available_rewards'] = list(zip(df[' unbiased_reward'], df[' biased_reward'])) 
        df["subject"] = n+1 # add subject id
        df = df[['subject','trial_number','available_rewards',' is_biased_choice', ' observed_reward']]  # Keep only columns with choices and outcomes
        df = df.rename(columns={'trial_number': 'trial',' is_biased_choice': 'choice', ' observed_reward': 'reward'})  # Rename the columns
        df.trial = df.trial + 1 # change indexing from 1 to 100
        df['choice'] = df['choice'].replace({' true': 1, ' false': 0}).astype(int) # 1 means chose the biased choice
        dataframes.append(df)   # Append the DataFrame to the list
    
    # how many subjects we load from
    Nsubject = n+1
    
    # Concatenate all DataFrames in the list into a single DataFrame
    df = pd.concat(dataframes, ignore_index=True)
    print("loaded data from ",Nsubject, " participants (", len(df), " trials in total)")
    
    return df, Nsubject

class RLModel(ABC):
    @abstractmethod
    def simulate(self, available_rewards):
        pass

    @abstractmethod
    def negLogLikelihood(self, pars, a, r):
        pass

    def fit(self, a, r):
        # get default values of parameters for each model
        x0 = self.initial_parameters()
        
        # get parameter bounds
        bounds = self.parameter_bounds()
        
        # run minimization of negative-LLH over bounded parameter space
        res = optimize.minimize(self.negLogLikelihood, args=(a, r), method='L-BFGS-B', x0=x0, bounds=bounds)
        
        # compute Bayesian Information Criterion
        bic = len(x0) * np.log(len(a)) + 2 * res.fun
        
        return bic, res.x, -res.fun  # BIC, parameters, log-likelihood

    @abstractmethod
    def initial_parameters(self):
        pass

    @abstractmethod
    def parameter_bounds(self):
        pass


class WinStayLoseSwitch(RLModel):
    def simulate(self, available_rewards, epsilon):
        # first choice is random
        ch = np.random.choice(2)
        a = [ch]
        
        # corresponding outcome
        oo = available_rewards[0,ch]
        r = [oo]

        T = available_rewards.shape[0] # number of trials
        p_all = np.empty((T,2))

        for t in range(T - 1):
            if oo == 1:  # win-stay
                p = [epsilon / 2, epsilon / 2]
                p[ch] = 1 - epsilon / 2
            else:  # lose-shift
                p = [1 - epsilon / 2, 1 - epsilon / 2]
                p[ch] = epsilon / 2
        
            p_all[t,:] = p
            
            # select action following these probabilities
            ch = np.random.choice(len(p), p=p)
        
            # corresponding outcome
            oo = available_rewards[t,ch]

            a.append(ch)
            r.append(oo)

        return np.array(a), np.array(r), p_all

    def negLogLikelihood(self, pars, a, r):
        epsilon = pars[0]
        choice_p = [0.5]

        for t in range(1, len(a)):
            if r[t-1] == 1:
                p = [epsilon / 2, epsilon / 2]
                p[a[t-1]] = 1 - epsilon / 2
            else:
                p = [1 - epsilon / 2, 1 - epsilon / 2]
                p[a[t-1]] = epsilon / 2

            choice_p.append(p[a[t]])

        return -np.sum(np.log(np.array(choice_p) + 1e-5))

    def initial_parameters(self):
        return [np.random.random()]

    def parameter_bounds(self):
        return [(0, 2)]


class RescorlaWagner(RLModel):
    def simulate(self, available_rewards, alpha, beta):
        # initial values for Q: 0.5 for each option
        Q = np.array([0.5, 0.5])
        a, r = [], []
    
        T = available_rewards.shape[0] # number of trials
        Qall = np.empty((T,2))

        # loop through all trials
        for t in range(T):
            # compute probability for selecting each option based on Q-values
            betaQ_rel = beta * Q - np.max(beta * Q)
            p = np.exp(betaQ_rel) / np.sum(np.exp(betaQ_rel))
    
            # select action following these probabilities
            ch = np.random.choice(len(p), p=p)
    
            # corresponding outcome
            oo = available_rewards[t,ch]
    
            # update Q-value for selected choice
            delta = oo - Q[ch] # prediction error
            Q[ch] += alpha * delta
            
            Qall[t,:] = Q

            # append to lists
            a.append(ch)
            r.append(oo)

        return np.array(a), np.array(r), Qall

    def negLogLikelihood(self, pars, a, r):
      alpha, beta = pars

      # initial values for Q
      Q = np.array([0.5, 0.5])
      choice_p = []
      T = len(a) # number of time steps

      #loop through all trials
      for t in range(T):
          # probability of choices
          betaQ_rel = beta * Q - np.max(beta * Q)
          p = np.exp(betaQ_rel) / np.sum(np.exp(betaQ_rel))
          choice_p.append(p[a[t]])

          # update Q-values based on outcome
          delta = r[t] - Q[a[t]]
          Q[a[t]] += alpha * delta

      #log-likelihood for individual trials (we add small values to avoid infinite if p is 0 - akin to including lapses)
      LLH = np.sum(np.log(np.array(choice_p) + 1e-5))
      return -LLH 

    def initial_parameters(self):
        return [np.random.random(), np.random.exponential()]

    def parameter_bounds(self):
        return [(0, 1), (0, np.inf)]


class ChoiceKernel(RLModel):
    def simulate(self, available_rewards, alpha, beta):
        CK = np.full(2, 0.001)
        a, r = [], []
        T = available_rewards.shape[0] # number of trials
        CKall = np.empty((T,2))

        for t in range(T):
            # compute probability for selecting each option based on Q-values
            betaCK_rel = beta * CK - np.max(beta * CK)
            p = np.exp(betaCK_rel) / np.sum(np.exp(betaCK_rel))
    
            # select action following these probabilities
            ch = np.random.choice(len(p), p=p)
    
            # corresponding outcome
            oo = available_rewards[t,ch]
            
            a.append(ch)
            r.append(oo)

            # update weight of selected choice
            CK *= (1 - alpha)
            CK[ch] += alpha
            
            CKall[t,:] = CK


        return np.array(a), np.array(r), CKall

    def negLogLikelihood(self, pars, a, r):
        alpha, beta = pars
        CK = np.full(2, 0.001)
        choice_p = []

        for t in range(len(a)):
            betaCK_rel = beta * CK - np.max(beta * CK)
            p = np.exp(betaCK_rel) / np.sum(np.exp(betaCK_rel))
            choice_p.append(p[a[t]])

            CK *= (1 - alpha)
            CK[a[t]] += alpha

        return -np.sum(np.log(np.array(choice_p) + 1e-5))

    def initial_parameters(self):
        return [np.random.random(), 0.5 + np.random.exponential()]

    def parameter_bounds(self):
        return [(0, 1), (0, np.inf)]


# Usage example:
# wsls = WinStayLoseShift()
# rw = RescorlaWagner()
# ck = ChoiceKernel()

# a, r = wsls.simulate(100, [0.7, 0.3], 0.1)
# bic, params, nll = wsls.fit(a, r)
