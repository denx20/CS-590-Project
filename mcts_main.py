from function import FunctionTerm, Function
from sequence_generator import make_n_random_functions
from mcts_networks import *
import numpy as np
import torch
#import pytorch_lightning as pl

import math
import random
from copy import deepcopy
import pickle
import argparse
import string
import os
import pandas as pd



LEVELS = 4

POSSIBLE_TERMS = [
    FunctionTerm(type="constant"),
    FunctionTerm(type="loc_term", exponent1=1),
    FunctionTerm(type="loc_term", exponent1=2),
    FunctionTerm(type="loc_term", exponent1=3),
]
for index_diff in range(1, 4):
    for power in range(1, 3):
        POSSIBLE_TERMS.append(FunctionTerm(type="power_term", exponent1=power,index_diff1=index_diff))


def generate_term_to_id_map():
    i = 0
    id_to_func_term = {}
    term_str_to_func_term = {}
    term_str_to_id = {}
    
    
    for t in POSSIBLE_TERMS:
        temp_f = Function()
        temp_f.addTerm(t)

        id_to_func_term[i] = t
        
        term_str = str(t).replace('0*','').replace('0','1')
        term_str_to_id[term_str] = i
        
        term_str_to_func_term[term_str] = t
        
        i += 1
            
    term_str_to_id['<ROOT>'] = i
    i += 1
    term_str_to_id['<EOS>'] = i
    return id_to_func_term, term_str_to_func_term, term_str_to_id


id_to_func_term, term_str_to_func_term, term_str_to_id = generate_term_to_id_map()
#print(term_str_to_id)
id_to_term_str = {v:k for k,v in term_str_to_id.items()}
#print(id_to_term_str)

TERM_TYPES = len(id_to_term_str)


def state_to_nn_input(s):
    terms_str = s.split('|')
    return torch.LongTensor([[term_str_to_id[t] for t in terms_str if t in term_str_to_id]])

def grid_search(sequence, terms, upper_bound = 5, lower_bound = -5):
    # sequence: target sequence
    # terms: list of all FunctionTerm objects proposed by MCTS
    
    coeff = sorted(list(range(1, upper_bound+1))+list(range(lower_bound,0)), key=lambda x: abs(x))
    base = len(coeff)
    digit_to_coeff = {i: coeff[i] for i in range(base)}


    def int_to_base_helper(num, base):
        ret = []
        while num > 0:
            ret.append(num % base)
            num = num // base
        return ret
    
    penalty = np.inf
    
    for i in range(base**len(terms)):
        term_coeffs = [digit_to_coeff[c] for c in int_to_base_helper(i, base)+[0]*50]
        f = Function()
        for j, term in enumerate(terms):
            term.updateCoeff(term_coeffs[j])
            f.addTerm(term)
        
        if f.startIndex() > len(sequence):
            continue
        
        targets = np.array(sequence[f.startIndex()-1:])
        predictions = [f.evaluate(sequence, n) for n in range(f.startIndex(), len(sequence)+1)]
        
        if None in predictions:
            raise Exception(f'None in prediction! Current f is f[n] = {f} and prediction is {predictions}')
        
        predictions = np.array(predictions)
        
        rmse_loss = np.sqrt(np.mean((predictions-targets)**2))
        if rmse_loss < penalty:
            penalty = rmse_loss
        
        del f
        
        if penalty == 0:
            return penalty  
        
    return penalty


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, vocab_size, target_seq, neural_network, max_depth=LEVELS, numMCTSSims=20, cpuct=1, 
                 reward=1, penalty=-1, coeff_upper_bound=5, coeff_lower_bound=-5):
        
        # MCTS hyperparameters
        self.numMCTSSims = numMCTSSims
        self.cpuct = cpuct
        
        self.vocab_size = vocab_size
        self.target_seq = target_seq
        self.max_depth = max_depth
        self.coeff_upper_bound = coeff_upper_bound
        self.coeff_lower_bound = coeff_lower_bound
        self.reward = reward
        self.penalty = penalty
        
        
        #self.nnet = nnet
        self.nn = neural_network
        
        #self.args = args
        self.Qsa = {}  # stores Q values for s,a , expected reward for taking action a from state s, key = (state,action)
        self.Nsa = {}  # stores number of times we take action a at state s, key = (state, action)
        self.Ns = {}  # stores #times board s was visited, key = state
        self.Ps = {}  # stores initial policy (returned by neural net), key = state, value = policy vector

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        
        self.terminal_rewards = {}
        
        self.correct_count = 0
    
        
    def getActionProb(self, s, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from node s.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.numMCTSSims):
            self.search(s)
        
        layer = s.count('|')
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.vocab_size)]
        
        if len(counts) == 0:
            print('This state has never been visited')
            return []

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / (counts_sum+1e-6) for x in counts]
        return np.array(probs)
    
    def getPolicy(self, s):
        policy = np.zeros(self.vocab_size)
        for a in range(self.vocab_size):
            if (s,a) in self.Nsa:
                policy[a] += self.Nsa[(s,a)]
        
        if policy.sum() > 0:
            return policy/policy.sum()
        else:
            return self.getActionProb(s)

    def search(self, s):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the game's
        outcome reward is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        
        Returns:
            v: the value of the current state s
        """

        
        #if s not in self.Es:
         #   self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        
        if s.endswith('<EOS>') or s.count('|') >= self.max_depth:
            # terminal node
            if s not in self.terminal_rewards:
                self.terminal_rewards[s] = self.getReward(s)
            return self.terminal_rewards[s]

        
        if s not in self.Ps:
            # leaf node
            
            states = state_to_nn_input(s)
            policy_estimate, v = self.nn(states, torch.Tensor(self.target_seq))
            self.Ps[s] = policy_estimate.detach().squeeze().numpy()
            v = float(v.detach())
            
            valids = self.getValidMoves(s)  # TODO: test this masking function
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= (np.sum(self.Ps[s])+1e-6)

            self.Vs[s] = valids
            self.Ns[s] = 0
            return v

        valids = self.Vs[s]
        if sum(valids) == 0:
            print(f'Error: valids at node {s} sum to 0')
        
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        ucb = np.ones(len(valids)) * -np.inf
        for a in range(len(valids)):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-6)  # Q = 0 ?

                #if u > cur_best:
                 #   cur_best = u
                  #  best_act = a
                ucb[a] = u
        
        
        #ucb = np.exp(ucb)
        #a = np.random.choice(len(valids), p=ucb/ucb.sum())
        a = ucb.argmax()
        next_s = s + '|' + id_to_term_str[a]

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v
    
    def getValidMoves(self, s):
        ret = np.zeros(self.vocab_size)
        terms = s.split('|')
        
        # If s reached max_depth, mark all moves as impossible
        if len(terms)-1 >= self.max_depth:
            ret[term_str_to_id['<EOS>']] = 1
            return ret
        
        for move in term_str_to_id:
            if move not in terms:
                ret[term_str_to_id[move]] = 1
        
        return ret
        
    def getReward(self, s, use_regression=True):
        # TODO: implement this method
        term_list = s.split('|')[1:]
        terms = []
        
        for t in term_list:
            if t != '<ROOT>' and t != '<EOS>':
                terms.append(term_str_to_func_term[t])
        
        if use_regression:
            start_index = 0
            for t in terms:
                t.updateCoeff(1)
                start_index = max(start_index, (t.index_diff1 or 0), (t.index_diff2 or 0))
            
            X = np.zeros((len(self.target_seq)-start_index, len(terms)))
            y = np.array(self.target_seq[start_index:])

            for i in range(len(y)):
                for j, term in enumerate(terms):
                    X[i,j] = term.evaluate(self.target_seq, start_index+i+1)

            beta = np.linalg.pinv(X).dot(y)
            beta = np.round_(beta)
            for i in range(len(beta)):
                if beta[i] > self.coeff_upper_bound:
                    beta[i] = self.coeff_upper_bound
                elif beta[i] < self.coeff_lower_bound:
                    beta[i] = self.coeff_lower_bound
                
            penalty = np.sum((y - X.dot(beta))**2)
        
        else:
            penalty = grid_search(self.target_seq, terms, upper_bound=self.coeff_upper_bound, 
                             lower_bound=self.coeff_lower_bound)
        
        if penalty == 0:
            print('Found a solution:', s)
            self.correct_count += 1
            return max(self.reward - len(terms), 1)
        
        else:
            return -np.sqrt(penalty+1)

class MCTSTrainer():

    def __init__(self, target_seq, vocab_size, numIters, numEpisodes, mcts_args, 
                 nn_args=None, neural_network=None, max_length=LEVELS, 
                 coeff_upper_bound=5, coeff_lower_bound=-5):
        
        if not nn_args and not neural_network:
            raise Exception('at least one of nn_args and neural_network should be provided')
            
        self.target_seq = target_seq
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        self.coeff_upper_bound = coeff_upper_bound
        self.coeff_lower_bound = coeff_lower_bound
        
        self.numIters = numIters
        self.numEpisodes = numEpisodes
        
        self.nn = neural_network
        if not neural_network:
            self.nn = MCTS_GRU(vocab_size, nn_args['embed_size'], nn_args['hidden_size'], 
                               nn_args['num_layers'], len(target_seq), nn_args['embedding_weights'])
            
        #self.opt = torch.optim.Adam(self.nn.parameters())
        
        self.mcts_args = mcts_args
        self.mcts = MCTS(vocab_size, target_seq, self.nn, max_depth=self.mcts_args['maxTreeLevel'], 
                             numMCTSSims=self.mcts_args['numMCTSSims'], cpuct=self.mcts_args['cpuct'])
        
        self.examples = []
        
        #self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        #self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self, root='<ROOT>', force_break=100):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        
        s = root   # initialise search tree
        examples = []
        intermediate_examples = []
        loop_count = 0
        
        while True:
            loop_count += 1
            
            if loop_count > force_break:
                print('While loop upper limit exceeded. Force break.')
                return examples
            
            # evaluate reward and return examples
            if s.endswith('<EOS>') or s.count('|') >= self.max_length:
                print('Evaluating terminal state...')
                if s in self.mcts.terminal_rewards:
                    reward = self.mcts.terminal_rewards[s]
                else:
                    reward = self.mcts.search(s)
                
                examples.append([s, self.mcts.getPolicy(s), reward])
                #examples += intermediate_examples
                return examples
            
            
            # update self.mcts.Qsa, Ns, Nsa, etc. by performing search() for numMCTSSims times
            self.mcts.getActionProb(s)
            
            # get MCTS policy at state s
            print('Getting MCTS policy for current episode...')
            policy = self.mcts.getPolicy(s)
            
            # find next move and append to s 
            if policy.sum() > 0:
                policy = policy/policy.sum() 
                #examples.append([s, policy])
                #intermediate_examples.append([s, policy, self.mcts.search(s)])
                examples.append([s, policy, self.mcts.search(s)])
                a = np.random.choice(len(policy), p=policy)    # sample action from improved policy
                
            else:
                uniform_policy = self.mcts.getValidMoves(s)
                a = np.random.choice(len(uniform_policy), p=uniform_policy/uniform_policy.sum())
            
            s += '|'+id_to_term_str[int(a)]
                                     
            
            

    def learn(self, reward=1, penalty=-1, num_train_epochs=1):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        
        # TODO: delete this later
        def isBetter(a,b):
            return True
        
        for i in range(self.numIters):
            print(f'=== Iter {i} ===')
            examples = []
            self.mcts = MCTS(self.vocab_size, self.target_seq, self.nn, max_depth=self.mcts_args['maxTreeLevel'], 
                             numMCTSSims=self.mcts_args['numMCTSSims'], cpuct=self.mcts_args['cpuct'], 
                             reward=reward, penalty=penalty)
            
            print(f'Running preliminary tree search on iter {i}...')
            self.mcts.getActionProb('<ROOT>')       # preliminary tree search
            
            for j in range(self.numEpisodes):
                print(f'Running Episode {j}...')
                examples += self.executeEpisode()         # collect training examples from MCTS
            
            # Add more examples
            #random_nodes = sorted(list(self.mcts.Ns.keys()), key=lambda k: -self.mcts.Ns[k])[:10]
            random_nodes = random.sample(list(self.mcts.Ns.keys()), min(10, len(self.mcts.Ns)))
            examples += [[node, self.mcts.getPolicy(node), self.mcts.search(node)] for node in random_nodes]
            
            examples += self.augmentExamples(examples)
            self.examples += examples
            
            print('Training neural network...')
            new_nn = self.trainNN(deepcopy(self.nn), examples, num_train_epochs)
            
            
            # if trained neural networks are better than old neural network, then replace old neural network
            if isBetter(new_nn, self.nn):   # TODO: define this function
                self.nn = new_nn
        
        
        
    def trainNN(self, neural_net, examples, num_train_epochs=10):
        # each element in examples has the form [state, Ps[s], z]
        # loss function = (v(s)-z)^2 - pi log p(s)
        
        def loss_fn(nn_value, nn_policy, mcts_value, mcts_policy):
            #print(mcts_value)
            #print(mcts_policy)
            ce_loss = torch.nn.CrossEntropyLoss()
            return (nn_value - mcts_value)**2 + ce_loss(nn_policy, mcts_policy)
        
        opt = torch.optim.Adam(neural_net.parameters())
        
        for _ in range(num_train_epochs):
            for example in examples:
                s = example[0]
                nn_policy, nn_value = neural_net(state_to_nn_input(s), torch.Tensor(self.target_seq))
                mcts_value = example[2]
                mcts_policy = example[1] #[self.mcts.Nsa[(s,a)] if (s,a) in self.mcts.Nsa else 0 for a in range(self.vocab_size)]
                
                if sum(mcts_policy) == 0:  # handle the case where no key (s,a) exists in self.mcts.Nsa
                    continue
                
                mcts_policy = torch.Tensor(mcts_policy)/(sum(mcts_policy))  # TODO: make this right
                mcts_policy = mcts_policy.reshape(1,-1)
                
                loss = loss_fn(nn_value, nn_policy, mcts_value, mcts_policy)
                #print('loss is', loss)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
        
        return neural_net
    
    def augmentExamples(self, examples):
        '''
        randomly permute a state's terms 
        '''
        new_examples = []
        for e in examples:
            state = e[0]
            terms = state.split('|')
            if terms[-1] == '<EOS>':
                terms.pop()
            
            if len(terms) > 1:
                terms = terms[1:]
                random.shuffle(terms)
                new_examples.append(['<ROOT>|'+'|'.join(terms), e[1], e[2]])
                
        return new_examples
            
        
    def getActualReward(self, s):
        return self.mcts.getReward(s)


def eval_expression_rmse(sequence, expression):
    candidates = []
    for t in expression.split('|'):
        if t != '<ROOT>' and t != '<EOS>':
            candidates.append(term_str_to_func_term[t])
    return grid_search(sequence, candidates)

def eval_nn(sequence, neural_network, root='<ROOT>'):
    print('sequence is: ', sequence)
    helper_mcts = MCTS(TERM_TYPES, None, None)
    temp_s = root
    
    rmse = 1e7
    minimum_rmse = 1e7
    best_state = None

    for _ in range(LEVELS+1):
        if temp_s.endswith('<EOS>'):
            break

        print('Proposal:', temp_s)
        rmse = eval_expression_rmse(sequence, temp_s)
        print('Is it correct? Loss =', rmse)

        if rmse < minimum_rmse:
            minimum_rmse = rmse
            best_state = temp_s

        reward_list = np.ones(TERM_TYPES) * -np.inf
        valids = helper_mcts.getValidMoves(temp_s)
        for a in range(TERM_TYPES):
            if valids[a]:
                temp_prob, temp_reward = neural_network(state_to_nn_input(temp_s+'|'+id_to_term_str[a]), torch.Tensor(sequence))
                reward_list[a] = temp_reward

        print('Reward:', temp_reward)
        print('-'*60)
        #temp_probs = coach.mcts.getValidMoves(temp_s)*temp_probs.detach().numpy()
        temp_s += '|'+id_to_term_str[int(reward_list.argmax())]

    print('Proposal:', temp_s)
    temp_probs, temp_reward = neural_network(state_to_nn_input(temp_s), torch.Tensor(sequence))
    print('Reward:', temp_reward)
    print('Is it correct? Loss =', eval_expression_rmse(sequence, temp_s))
    print('-'*60)
    
    return minimum_rmse, best_state


def save_results(tag, avg_rmse, correct_count, term_types, nterms, model_type, use_hint, nn_args, 
                numMCTSSims, depth_limit, mcts_cpuct, iters, episodes, reward):
    if os.path.isfile('mcts_experiment_results.csv'):
        df = pd.read_csv('mcts_experiment_results.csv', index_col=0)
    else:
        df = pd.DataFrame(
                columns=['tag', 'avg_rmse', 'correct_count', 'term_types', 'nterms', 'model', 'nn_hyperparams', 'use_hint', 'numMCTSSims', 'depth_limit', 'mcts_cpuct', 'iters', 'episodes', 'reward']
            )
    
    new_df = {
        'tag': tag,
        'avg_rmse': avg_rmse, 
        'correct_count': correct_count,
        'term_types': term_types, 
        'nterms': nterms, 
        'model': model_type, 
        'nn_hyperparams': str(nn_args), 
        'use_hint': use_hint, 
        'numMCTSSims': numMCTSSims, 
        'depth_limit': depth_limit, 
        'mcts_cpuct': mcts_cpuct,
        'iters': iters, 
        'episodes': episodes, 
        'reward': reward
    }

    df = df.append(new_df, ignore_index = True)
    df.to_csv('mcts_experiment_results.csv')
    

def load_data(nterms, train_data=True):
    # return a list [sequence, mask] elements
    if train_data:
        df = pd.read_csv(f'data/train/{nterms}/{nterms}.csv', names=['prompt', 'completion'], delimiter='],', engine='python')
    else:
        df = pd.read_csv(f'data/test/{nterms}/{nterms}.csv', names=['prompt', 'completion'], delimiter='],', engine='python')
    
    data = []
    for i in range(len(df)):
        seq = df.loc[i, 'prompt']
        seq = seq.replace('[','').replace(']','')
        seq = seq.split(',')
        seq = [int(s) for s in seq]

        mask =df.loc[i, 'completion']
        mask = mask.replace('[','').replace(']','')
        mask = mask.split(',')
        mask = [eval(s) for s in mask]

        data.append([seq, mask])
    
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nterms', type=int, help='number of terms in ground-truth function', default=2)
    parser.add_argument('--mctssims', type=int, help='number of MCTS Simulations to run to get action probability', default=1000)
    parser.add_argument('--model_type', type=str, help='model type, MLP or GRU', default="MLP")
    parser.add_argument('--hint', type=bool, help='whether to use hint to help guide MCTS', default=False)

    input_args = parser.parse_args()

    rand_tag = "".join(list(np.random.choice(list(string.ascii_lowercase), 10)))
    print('Experiment tag =', rand_tag)
    print('Number of possible terms =', TERM_TYPES)
    for t in POSSIBLE_TERMS:
        print(t)

    nterms = input_args.nterms
    assert nterms <= LEVELS, 'nterms must not exceed max depth!'
    print('Number of terms in ground-truth expression =', nterms)

    '''
    np.random.seed(590)
    random.seed(590)
    data = make_n_random_functions(
        n=1000,
        sequence_bound=1000,
        nterms=nterms,
        coefficient_range=(-5, 5),
        initial_terms_range=(1, 3)
    )

    random.shuffle(data)
    N_train = int(0.8*len(data))
    train_data, test_data = data[:N_train], data[N_train:]
    '''

    '''
    with open(f'mcts_train_data_{nterms}terms.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(f'mcts_test_data_{nterms}terms.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    '''

    '''
    # Load previously generated data
    with open(f'mcts_train_data_{nterms}terms.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    with open(f'mcts_test_data_{nterms}terms.pkl', 'rb') as f:
        test_data = pickle.load(f)
    '''

    # load data
    train_data = load_data(nterms, train_data=True)
    test_data = load_data(nterms, train_data=False)

    train_sequence_list = [d[0] for d in train_data]
    test_sequence_list = [d[0] for d in test_data]
    sequence_length = len(train_sequence_list[0])
    

    # Experiment args
    # MCTS args
    mcts_args = {
        'maxTreeLevel': LEVELS,
        'numMCTSSims': input_args.mctssims,
        'cpuct': 1   
    }
    reward = 10
    print('MCTS parameters:')
    print(mcts_args)
    print('reward =', reward)
    print('')

    # neural network args
    model_type = input_args.model_type
    nn_args = {
        'embed_size': 20,
        'hidden_size': 40,
        'num_layers': 3
    }
    print('Neural Network parameters:')
    print(f"architecture={model_type}, embed_size={nn_args['embed_size']}, hidden_size={nn_args['hidden_size']}, num_layers={nn_args['num_layers']}")
    print('')
    if model_type == 'MLP':
        model = MCTS_MLP(TERM_TYPES, nn_args['embed_size'], nn_args['hidden_size'], nn_args['num_layers'], sequence_length)
    elif model_type == 'GRU':
        model = MCTS_GRU(TERM_TYPES, nn_args['embed_size'], nn_args['hidden_size'], nn_args['num_layers'], sequence_length)
    elif model_type == 'Transformer':
        model = MCTS_Transformer()
    else:
        raise NotImplementedError(f'unrecognized model type {model_type}')

    # MCTS trainer args
    numIters = 3
    numEpisodes = 4
    use_hint = input_args.hint
    print('Neural MCTS training parameters:')
    print(f'# of iterations={numIters}, # of episodes per iteration={numEpisodes}')
    print('Using hint?', use_hint)

    print('='*30 + 'Training begins now' + '='*30)
    all_examples = []

    # Training
    for i, seq in enumerate(train_sequence_list):
        print("Sequence", i)
        print("Sequence is:", seq)
        trainer = MCTSTrainer(seq, TERM_TYPES, numIters, numEpisodes, mcts_args, neural_network=model)
        if use_hint:
            ground_truth_terms = np.array(train_data[i][1])
            hint_term_id = np.random.choice(len(POSSIBLE_TERMS), p=ground_truth_terms/ground_truth_terms.sum())
            print(f'Hinting MCTS to search node "<ROOT>|{id_to_term_str[hint_term_id]}"')
            trainer.executeEpisode(root='<ROOT>|'+id_to_term_str[hint_term_id])
        
        trainer.learn(reward=reward, num_train_epochs=1)
        for e in trainer.examples:
            all_examples.append([seq]+e)
        model = trainer.nn

    print("Number of training examples collected =", len(all_examples))
    #with open('mcts_training_examples.pkl', 'wb') as f:
    #    pickle.dump(all_examples, f)

    # Save model
    torch.save(model, f'mcts_models/mcts_model_{rand_tag}.pt')

    print('='*30 + 'Evaluation begins now' + '='*30)
    rmse_list = []
    best_state_list = []
    perfect_counts = 0
    # Evaluation
    for seq in test_sequence_list:
        rmse, best_state = eval_nn(seq, trainer.nn)
        rmse_list.append(rmse)
        best_state_list.append(best_state)
        if rmse == 0:
            perfect_counts += 1
    
    avg_rmse = sum(rmse_list)/len(rmse_list)
    print('Mean RMSE on test data:', avg_rmse)
    print('Number of perfectly solved examples:', perfect_counts)

    # save experiment run results to csv file
    save_results(rand_tag, avg_rmse, perfect_counts, TERM_TYPES, nterms, model_type, use_hint, nn_args, 
                mcts_args['numMCTSSims'], mcts_args['maxTreeLevel'], mcts_args['cpuct'], numIters, numEpisodes, reward)
    

    
    