import torch
import pandas as pd
import numpy as np
import argparse
from function import FunctionTerm, Function
from mcts_main import eval_nn, load_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='model weight file name')
    parser.add_argument('trained_nterms', type=int, help='number of terms per function in test set')

    input_args = parser.parse_args()
    model_name = input_args.model_name
    trained_nterms = input_args.trained_nterms

    model = torch.load(f'mcts_models/mcts_model_{model_name}.pt')
    model.eval()

    for nterms in range(trained_nterms+1, 6):
        print(f'Now testing generalization of model {model_name} to nterms = {nterms}...')

        test_data = load_data(nterms, train_data=False)
        test_sequence_list = [d[0] for d in test_data]

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
