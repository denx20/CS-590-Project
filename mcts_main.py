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


random.seed(590)

# 'loc_term', 'power_term', 'interaction_term'

LEVELS = 4
TREE_LEVELS = [
    [FunctionTerm('constant')],
    [
      FunctionTerm('loc_term'), 
      FunctionTerm('power_term', index_diff1=1), 
      FunctionTerm('power_term', index_diff1=2), 
      FunctionTerm('power_term', index_diff1=3)
    ],
    [
      FunctionTerm('loc_term', exponent1=2), 
      FunctionTerm('power_term', exponent1=2, index_diff1=1), 
      FunctionTerm('power_term', exponent1=2, index_diff1=2), 
      FunctionTerm('power_term', exponent1=2, index_diff1=3)
    ],
    #[
    # FunctionTerm('loc_term', exponent1=2), 
    # FunctionTerm('power_term', exponent1=2, index_diff1=1), 
    # FunctionTerm('power_term', exponent1=2, index_diff1=2), 
    # FunctionTerm('power_term', exponent1=2, index_diff1=3)
    #],
    [
      FunctionTerm('loc_term', exponent1=3), 
      FunctionTerm('power_term', exponent1=3, index_diff1=1), 
      FunctionTerm('power_term', exponent1=3, index_diff1=2), 
      FunctionTerm('power_term', exponent1=3, index_diff1=3)
    ],
    [
      FunctionTerm('interaction_term', exponent1=1, exponent2=1, index_diff1=1,index_diff2=2), 
      FunctionTerm('interaction_term', exponent1=1, exponent2=1, index_diff1=1,index_diff2=3), 
      FunctionTerm('interaction_term', exponent1=1, exponent2=1, index_diff1=2,index_diff2=3)
    ],
    [
      FunctionTerm('interaction_term', exponent1=2, exponent2=1, index_diff1=1, index_diff2=2), 
      FunctionTerm('interaction_term', exponent1=2, exponent2=1, index_diff1=1, index_diff2=3), 
      FunctionTerm('interaction_term', exponent1=2, exponent2=1, index_diff1=2, index_diff2=3), 
      FunctionTerm('interaction_term', exponent1=1, exponent2=2, index_diff1=1, index_diff2=2), 
      FunctionTerm('interaction_term', exponent1=1, exponent2=2, index_diff1=1, index_diff2=3), 
      FunctionTerm('interaction_term', exponent1=1, exponent2=2, index_diff1=2, index_diff2=3)
    ]
]


def generate_term_to_id_map(level):
    i = 0
    id_to_func_term = {}
    term_str_to_func_term = {}
    term_str_to_id = {}
    
    for level in TREE_LEVELS[:level]:
        for t in level:
            temp_f = Function()
            temp_f.addTerm(t)
            #key = list(temp_f.terms.keys())[0]
            #term_to_id[key] = i
            
            id_to_func_term[i] = t
            
            term_str = str(t).replace('0*','').replace('0','1')
            term_str_to_id[term_str] = i
            
            term_str_to_func_term[term_str] = t
            
            i += 1
            
    term_str_to_id['<ROOT>'] = i
    i += 1
    term_str_to_id['<EOS>'] = i
    return id_to_func_term, term_str_to_func_term, term_str_to_id


id_to_func_term, term_str_to_func_term, term_str_to_id = generate_term_to_id_map(LEVELS)
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

    def __init__(self, vocab_size, target_seq, neural_network, max_depth=len(TREE_LEVELS), numMCTSSims=20, cpuct=1, 
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
        
    def getReward(self, s):
        # TODO: implement this method
        term_list = s.split('|')[1:]
        terms = []
        
        for t in term_list:
            if t != '<ROOT>' and t != '<EOS>':
                terms.append(term_str_to_func_term[t])
        
        
        #len(term_list)-2
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
                 nn_args=None, neural_network=None, max_length=len(TREE_LEVELS), 
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
    
    for _ in range(10):
        if temp_s.endswith('<EOS>') or rmse <= 1:
            break

        print('Proposal:', temp_s)
        rmse = eval_expression_rmse(sequence, temp_s)
        print('Is it correct? Loss =', rmse)

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


if __name__ == '__main__':
    print('Number of terms =',TERM_TYPES)

    data = make_n_random_functions(
        n=1000,
        sequence_bound=1000,
        nterms=2,
        coefficient_range=(-5, 5),
        initial_terms_range=(1, 3)
    )

    sequence_list = [d[0] for d in data]
    sequence_length = len(sequence_list[0])

    # MCTS args
    mcts_args = {
        'maxTreeLevel': 4,
        'numMCTSSims': 1000,
        'cpuct': 1   
    }
    reward = 10

    # neural network args
    embed_size = 20
    hidden_size = 30 
    num_layers = 3

    # MCTS trainer args
    numIters = 5
    numEpisodes = 4


    #model = MCTS_MLP(TERM_TYPES, embed_size, hidden_size, num_layers, sequence_length) 
    model = MCTS_GRU(TERM_TYPES, embed_size, hidden_size, num_layers, sequence_length)

    all_examples = []

    for i, seq in enumerate(sequence_list[:1]):
        print("Sequence", i)
        trainer = MCTSTrainer(seq, TERM_TYPES, numIters, numEpisodes, mcts_args, neural_network=model)
        trainer.learn(reward=reward, num_train_epochs=1)
        for e in trainer.examples:
            all_examples.append([seq]+e)
        model = trainer.nn

    print(len(all_examples))
    print(all_examples[0])

    
    