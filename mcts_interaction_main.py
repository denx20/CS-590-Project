from function import FunctionTerm, Function
from mcts_networks import *
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import math
import random
from copy import deepcopy
import pickle
import argparse
import string
import os
import pandas as pd
from collections import defaultdict


BATCH_SIZE = 256
cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")

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

for index_diff1 in range(1, 3):
    for index_diff2 in range(index_diff1+1, 4):
        for exponent1 in range(1, 3):
            for exponent2 in range(1, 3):
                POSSIBLE_TERMS.append(
                    FunctionTerm(
                        type="interaction_term",
                        exponent1=exponent1,
                        exponent2=exponent2,
                        index_diff1=index_diff1,
                        index_diff2=index_diff2,
                    )
                )


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
    return torch.LongTensor([term_str_to_id[t] for t in terms_str if t in term_str_to_id])

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
            
            states = state_to_nn_input(s).reshape(1,-1)
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
                
            penalty = np.sqrt(np.sum((y - X.dot(beta))**2))
        
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

    def __init__(self, target_seq, vocab_size, numIters, numEpisodes, mcts_args, opt=None, scheduler=None,
                 nn_args=None, neural_network=None, train_nn=True, max_length=LEVELS, 
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

        self.train_nn = train_nn
            
        self.opt = opt
        self.scheduler = scheduler
        if self.opt is None:
            self.opt = torch.optim.Adam(self.nn.parameters())
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, patience=20)
        
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
            
            # sample intermediate nodes with <EOS> token appended
            examples += [[e[0]+'|<EOS>', np.zeros(self.vocab_size), self.getActualReward(e[0]+'|<EOS>')] for e in examples if e[-5:]!='<EOS>']
            
            # sample node permutations
            examples += self.augmentExamples(examples)
            self.examples += examples
            
            if self.train_nn:
                print('Training neural network...')
                self.trainNN(examples, num_train_epochs)
            
            
            # if trained neural networks are better than old neural network, then replace old neural network
            #if isBetter(new_nn, self.nn):   # TODO: define this function
            #    self.nn = new_nn
        
        
    def trainNN(self, examples, num_train_epochs=1):
        # each element in examples has the form [state, Ps[s], z]
        # loss function = (v(s)-z)^2 - pi log p(s)
        
        def loss_fn(nn_value, nn_policy, mcts_value, mcts_policy):
            #print(mcts_value)
            #print(mcts_policy)
            ce_loss = torch.nn.CrossEntropyLoss()
            return (nn_value - mcts_value)**2 + ce_loss(nn_policy, mcts_policy)
        
        # opt = torch.optim.Adam(neural_net.parameters())
        
        for _ in range(num_train_epochs):
            avg_loss = 0
            for example in examples:
                s = example[0]
                nn_policy, nn_value = self.nn(state_to_nn_input(s).reshape(1,-1), torch.Tensor(self.target_seq))
                mcts_value = example[2]
                mcts_policy = example[1] #[self.mcts.Nsa[(s,a)] if (s,a) in self.mcts.Nsa else 0 for a in range(self.vocab_size)]
                
                if sum(mcts_policy) == 0:  # handle the case where no key (s,a) exists in self.mcts.Nsa
                    continue
                
                mcts_policy = torch.Tensor(mcts_policy)/(sum(mcts_policy)+1e-7)  # TODO: make this right
                mcts_policy = mcts_policy.reshape(1,-1)
                
                loss = loss_fn(nn_value, nn_policy, mcts_value, mcts_policy)
                avg_loss += loss.item()
                if loss.isnan().any():
                    print('ERROR! Loss is nan! \n')
                    print(example)
                    print(mcts_policy)
                    print(mcts_value)
                    print(nn_policy)
                    print(nn_value)
                    raise Exception('ERROR! Loss is nan! \n')
                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            
            print('Training loss =', avg_loss/(len(examples)+1e-7))

    
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

                for _ in range(len(terms)):
                    random.shuffle(terms)
                    new_examples.append(['<ROOT>|'+'|'.join(terms), e[1], e[2]])
                
        return new_examples
            
        
    def getActualReward(self, s):
        return self.mcts.getReward(s)


class MCTSDataset(torch.utils.data.Dataset):
    def __init__(self, training_examples):
        # each example in training_examples has the format [sequence (list or np.ndarray), node (str), MCTS policy]
        self.data = [
            {
            'seq': torch.Tensor(e[0]), 
            'node': state_to_nn_input(e[1]),
            'policy': torch.Tensor(e[2]),
            'reward': torch.Tensor([e[3]])
            } 
            for e in training_examples
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ByLengthSampler(torch.utils.data.BatchSampler):
    """
    Allows for sampling minibatches of examples all of the same sequence length;
    adapted from https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/13.
    """
    def __init__(self, dataset, key, batchsize, shuffle=True, drop_last=False):
        # import ipdb
        # ipdb.set_trace()
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seqlens = torch.LongTensor([example[key].size(-1) for example in dataset])
        self.nbatches = len(self._generate_batches())

    def _generate_batches(self):
        # shuffle examples
        seqlens = self.seqlens
        perm = torch.randperm(seqlens.size(0)) if self.shuffle else torch.arange(seqlens.size(0))
        batches = []
        len2batch = defaultdict(list)
        for i, seqidx in enumerate(perm):
            seqlen, seqidx = seqlens[seqidx].item(), seqidx.item()
            len2batch[seqlen].append(seqidx)
            if len(len2batch[seqlen]) >= self.batchsize:
                batches.append(len2batch[seqlen][:])
                del len2batch[seqlen]
                
        # add any remaining batches
        if not self.drop_last:
            for length, batchlist in len2batch.items():
                if len(batchlist) > 0:
                    batches.append(batchlist)
        # shuffle again so we don't always start w/ the most common sizes
        batchperm = torch.randperm(len(batches)) if self.shuffle else torch.arange(len(batches))
        return [batches[idx] for idx in batchperm]

    def batch_count(self):
        return self.nbatches

    def __len__(self):
        return len(self.seqlens)

    def __iter__(self):
        batches = self._generate_batches()
        for batch in batches:
            yield batch

def collate(batchdictseq):
    batch_seqs = torch.stack([batchdictseq[i]['seq'] for i in range(len(batchdictseq))])
    batch_nodes = torch.stack([batchdictseq[i]['node'] for i in range(len(batchdictseq))])
    batch_policys = torch.stack([batchdictseq[i]['policy'] for i in range(len(batchdictseq))])
    batch_rewards = torch.stack([batchdictseq[i]['reward'] for i in range(len(batchdictseq))])
    
    return batch_seqs, batch_nodes, batch_policys, batch_rewards


def filter_mcts_examples(examples, filter_eos_ratio=0.1):
    ret = []
    eos_examples = []
    positive_count = 0
    negative_eos_count = 0
    for e in examples:
        if e[3] > 0:
            positive_count += 1
        if e[3] < 0 and e[1][-5:] == '<EOS>':
            negative_eos_count += 1
        if e[1][-5:] == '<EOS>' and e[3] > 0:
            eos_examples.append(e)
            continue
        ret.append(e)
    
    print(f'Number of positive examples = {positive_count}, Number of negative examples = {len(examples) - positive_count}')
    print(f'Number of positive EOS examples = {len(eos_examples)}, Number of negative EOS examples = {negative_eos_count}')

    random.shuffle(eos_examples)
    eos_retain = int(len(eos_examples) * filter_eos_ratio)
    print(f'Filtered out {len(eos_examples) - eos_retain} examples ending with <EOS> tokens')
    return ret + eos_examples[:eos_retain]


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
                temp_prob, temp_reward = neural_network(state_to_nn_input(temp_s+'|'+id_to_term_str[a]).reshape(1,-1), torch.Tensor(sequence))
                reward_list[a] = temp_reward

        print('Reward:', temp_reward)
        print('-'*60)
        #temp_probs = coach.mcts.getValidMoves(temp_s)*temp_probs.detach().numpy()
        temp_s += '|'+id_to_term_str[int(reward_list.argmax())]

        if temp_s.endswith('<EOS>') or minimum_rmse == 0:
            break

    print('Proposal:', temp_s)
    temp_probs, temp_reward = neural_network(state_to_nn_input(temp_s).reshape(1,-1), torch.Tensor(sequence))
    print('Reward:', temp_reward)
    print('Is it correct? Loss =', eval_expression_rmse(sequence, temp_s))
    print('-'*60)
    
    return minimum_rmse, best_state


def save_results(csv_name, tag, avg_rmse, correct_count, term_types, nterms, model_type, use_hint, nn_args, 
                numMCTSSims, depth_limit, mcts_cpuct, iters, episodes, reward, epochs, outer_iters=None):
    if os.path.isfile(csv_name):
        df = pd.read_csv(csv_name, index_col=0)
    else:
        df = pd.DataFrame(
                columns=['tag', 'term_types', 'nterms', 'avg_rmse', 'correct_count', 'model', 'nn_hyperparams', 'epochs', 'outer_iters',
                'use_hint', 'numMCTSSims', 'depth_limit', 'mcts_cpuct', 'mcts_iters', 'mcts_episodes', 'reward']
            )
    
    new_df = {
        'tag': tag,
        'term_types': term_types, 
        'nterms': nterms, 
        'avg_rmse': avg_rmse, 
        'correct_count': correct_count,
        'model': model_type, 
        'nn_hyperparams': str(nn_args), 
        'epochs': epochs, 
        'outer_iters': outer_iters,
        'use_hint': use_hint, 
        'numMCTSSims': numMCTSSims, 
        'depth_limit': depth_limit, 
        'mcts_cpuct': mcts_cpuct,
        'mcts_iters': iters, 
        'mcts_episodes': episodes, 
        'reward': reward
    }

    df = df.append(new_df, ignore_index = True)
    df.to_csv(csv_name)
    

def load_data(nterms, train_data=True):
    # return a list [sequence, mask] elements
    if train_data:
        df = pd.read_csv(f'data/train/{nterms}/{nterms}_int.csv', names=['prompt', 'completion'], delimiter='],', engine='python')
    else:
        df = pd.read_csv(f'data/test/{nterms}/{nterms}_int.csv', names=['prompt', 'completion'], delimiter='],', engine='python')
    
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
    parser.add_argument('--model_type', type=str, help='model type, MLP or GRU or TFR', default="MLP")
    parser.add_argument('--outer_iters', type=int, help='Number of outer iterations', default=5)
    parser.add_argument('--dim', type=int, help='neural network hidden dimension and attention dimension', default=256)
    parser.add_argument('--layers', type=int, help='neural network depth', default=4)
    parser.add_argument('--hint', type=bool, help='whether to use hint to help guide MCTS', default=False)
    parser.add_argument('--collect_first', type=bool, help='whether to collect all training examples first', default=True)
    parser.add_argument('--test_code', type=bool, help='testing', default=False)

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
    with open(f'mcts_train_data_{nterms}terms.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(f'mcts_test_data_{nterms}terms.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    '''

    # load data
    train_data = load_data(nterms, train_data=True)
    test_data = load_data(nterms, train_data=False)

    train_sequence_list = [d[0] for d in train_data] 
    test_sequence_list = [d[0] for d in test_data]
    
    if input_args.test_code:
        print('THIS IS A TEST RUN, RESULTS WILL NOT BE SAVED')
        train_sequence_list = train_sequence_list[:10]
        test_sequence_list = test_sequence_list[:10]

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
        'embed_size': input_args.dim,
        'hidden_size': input_args.dim,
        'num_layers': 4,
        'encoder_attn': True,
        'encoder_append_window': True  # This matters only when encoder_attn is set to False
    }
    print('Neural Network parameters:')
    print(f"architecture={model_type}, embed_size={nn_args['embed_size']}, hidden_size={nn_args['hidden_size']}, num_layers={nn_args['num_layers']}")
    print(nn_args)
    print('')
    if model_type == 'MLP':
        model = MCTS_MLP(TERM_TYPES, nn_args['embed_size'], nn_args['hidden_size'], nn_args['num_layers'], sequence_length, 
                        encoder_attn=nn_args['encoder_attn'], encoder_append_window=nn_args['encoder_append_window'])
    elif model_type == 'GRU':
        model = MCTS_GRU(TERM_TYPES, nn_args['embed_size'], nn_args['hidden_size'], nn_args['num_layers'], sequence_length, 
                        encoder_attn=nn_args['encoder_attn'], encoder_append_window=nn_args['encoder_append_window'])
    elif model_type == 'TFR':
        model = MCTS_Transformer(TERM_TYPES, nn_args['embed_size'], nn_args['hidden_size'], nn_args['num_layers'], sequence_length)
        del nn_args['encoder_attn']
        del nn_args['encoder_append_window']
    elif 'checkpoint-' in model_type:
        model_tag = model_type.split('-')[-1]
        model = torch.load(f'mcts_models/mcts_model_{model_tag}.pt')
    else:
        raise NotImplementedError(f'unrecognized model type {model_type}')
    print(model)

    # MCTS trainer args
    numIters = 3
    numEpisodes = 4
    use_hint = input_args.hint
    print(f'Neural MCTS training parameters:\n# of iterations={numIters}, # of episodes per iteration={numEpisodes}\nUsing hint? {use_hint}')

    print('='*30 + 'Training begins now' + '='*30)
    # Training
    collect_first = input_args.collect_first
    epochs = 20
    outer_iters = input_args.outer_iters
    learning_rate_init = 5*1e-5
    print('Initial learning rate =', learning_rate_init)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10)

    if collect_first:
        print('Collect examples first and train later')
        print('Number of outer iterations =', outer_iters)
        print('Number of training epochs per outer iteration =', epochs)

        temp_filename = f'mcts_model_temporary_{rand_tag}.pt'
        best_valid_loss = float('inf')

        mse_loss = torch.nn.MSELoss()
        bce_loss = torch.nn.BCELoss()
        def loss_fn(nn_reward, nn_policy, mcts_reward, mcts_policy):
            return mse_loss(nn_reward, mcts_reward) + bce_loss(nn_policy, mcts_policy)

        for iter in range(outer_iters):
            print('Outer Iter', iter+1)
            all_examples = []

            # collect examples first
            model.eval()
            model = model.to(cpu_device)
            for i, seq in enumerate(train_sequence_list):
                print("Sequence", i)
                print("Sequence is:", seq)
                # TODO: modify MCTSTrainer to include train_model = True/False option
                trainer = MCTSTrainer(seq, TERM_TYPES, numIters, numEpisodes, mcts_args, neural_network=model, train_nn=False)
                if use_hint:
                    ground_truth_terms = np.array(train_data[i][1])
                    hint_term_id = np.random.choice(len(POSSIBLE_TERMS), p=ground_truth_terms/ground_truth_terms.sum())
                    print(f'Hinting MCTS to search node "<ROOT>|{id_to_term_str[hint_term_id]}"')
                    trainer.executeEpisode(root='<ROOT>|'+id_to_term_str[hint_term_id])
                
                trainer.learn(reward=reward)
                for e in trainer.examples:
                    all_examples.append([seq]+e)
            print("Number of training examples collected =", len(all_examples))

            # Check data imbalance
            positive_examples_count = 0
            positive_eos_examples_count = 0
            negative_eos_examples_count = 0
            for e in all_examples:
                if e[3] > 0:
                    positive_examples_count += 1
                    if e[1][-5:] == '<EOS>':
                        positive_eos_examples_count += 1
                elif e[1][-5:] == '<EOS>': 
                    negative_eos_examples_count += 1
            
            print(f'Number of positive examples = {positive_examples_count}, Number of negative examples = {len(all_examples) - positive_examples_count}')
            print(f'Number of positive EOS examples = {positive_eos_examples_count}, Number of negative EOS examples = {negative_eos_examples_count}')

            # train model
            random.shuffle(all_examples)
            train_dataset = MCTSDataset(all_examples[:int(len(all_examples)*0.9)])
            train_loader = DataLoader(train_dataset,
                            batch_sampler=ByLengthSampler(train_dataset, key='node', batchsize=BATCH_SIZE, shuffle=True), 
                            collate_fn=collate, num_workers=2)
            
            valid_dataset = MCTSDataset(all_examples[int(len(all_examples)*0.9):])
            valid_loader = DataLoader(valid_dataset,
                            batch_sampler=ByLengthSampler(valid_dataset, key='node', batchsize=BATCH_SIZE, shuffle=True), 
                            collate_fn=collate, num_workers=2)

            model = model.to(cuda_device)

            for i in range(epochs):
                # train step
                model.train()
                total_train_loss = 0
                train_count = 0
                for X_seq, X_node, y_policy, y_reward in train_loader:
                    opt.zero_grad()

                    pred_policy, pred_reward = model(X_node.to(cuda_device), X_seq.to(cuda_device))
                    loss = loss_fn(pred_reward, pred_policy, y_reward.to(cuda_device), y_policy.to(cuda_device))

                    total_train_loss += loss.item()
                    train_count += len(X_seq)

                    loss.backward()
                    opt.step()
                
                # valid step
                model.eval()
                total_valid_loss = 0
                valid_count = 0
                with torch.no_grad():
                    for X_seq, X_node, y_policy, y_reward in valid_loader:
                        pred_policy, pred_reward = model(X_node.to(cuda_device), X_seq.to(cuda_device))
                        curr_valid_loss = loss_fn(pred_reward, pred_policy, y_reward.to(cuda_device), y_policy.to(cuda_device))

                        total_valid_loss += curr_valid_loss.item()
                        valid_count += len(X_seq)
                
                epoch_train_loss = total_train_loss/(1e-7 + train_count)
                epoch_valid_loss = total_valid_loss/(1e-7 + valid_count)

                if scheduler is not None:
                    scheduler.step(epoch_valid_loss)
                
                print(f'Epoch {i+1}: training Loss = {epoch_train_loss}, validation loss = {epoch_valid_loss}') 

                if (epoch_valid_loss < best_valid_loss):
                    best_valid_loss = epoch_valid_loss
                    ## Save the current model
                    torch.save(model, temp_filename)
            
        # load the best model during training
        model = torch.load(temp_filename)
        os.remove(temp_filename)

    else:
        print('Collect examples and train simultaneously')
        all_examples = []
        for i, seq in enumerate(train_sequence_list):
            print("Sequence", i)
            print("Sequence is:", seq)
            trainer = MCTSTrainer(seq, TERM_TYPES, numIters, numEpisodes, mcts_args, opt=opt, scheduler=scheduler, neural_network=model, 
                                train_nn=True)
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

    if scheduler is not None:
        print("Final learning rate is", opt.param_groups[0]['lr'])
    #with open('mcts_training_examples.pkl', 'wb') as f:
    #    pickle.dump(all_examples, f)


    # Evaluation
    print('='*30 + 'Evaluation begins now' + '='*30)
    model.eval()
    model.to(cpu_device)

    rmse_list = []
    best_state_list = []
    perfect_counts = 0

    for seq in test_sequence_list:
        rmse, best_state = eval_nn(seq, model)
        rmse_list.append(rmse)
        best_state_list.append(best_state)
        if rmse == 0:
            perfect_counts += 1
    
    avg_rmse = sum(rmse_list)/len(rmse_list)
    print('Mean RMSE on test data:', avg_rmse)
    print('Number of perfectly solved examples:', perfect_counts)

    if not input_args.test_code:
        # Save model
        if not os.path.isdir('mcts_models'):
            os.mkdir('mcts_models')
        torch.save(model, f'mcts_models/mcts_model_{rand_tag}.pt')

        # save experiment run results to csv file
        csv_filename = 'mcts_experiment_results_new.csv'
        save_results(csv_filename, rand_tag, avg_rmse, perfect_counts, TERM_TYPES, nterms, model_type, use_hint, nn_args, 
                    mcts_args['numMCTSSims'], mcts_args['maxTreeLevel'], mcts_args['cpuct'], numIters, numEpisodes, reward, epochs, outer_iters)
    
    

    
    