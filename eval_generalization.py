import torch
import pandas as pd
import numpy as np
import argparse
from function import FunctionTerm, Function
import mcts_main
import mcts_interaction_main



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='model weight file name')
    parser.add_argument('--csv_filepath', type=str, help='filepath of csv file containing experiment results', default='mcts_experiment_results_new.csv')

    input_args = parser.parse_args()
    model_name = input_args.model_name
    csv_filepath = input_args.csv_filepath
    df = pd.read_csv(csv_filepath, index_col=0)
    row = df[df['tag'] == model_name]
    if len(row) == 0:
        raise Exception(f'Model {model_name} not found in csv file')
    
    trained_nterms = row['nterms'].values[0]
    term_types = row['term_types'].values[0]
    original_correct_count = row['correct_count'].values[0]
    print('Number of term types =', term_types)
    print('Original correct count =', original_correct_count)

    model = torch.load(f'mcts_models/mcts_model_{model_name}.pt')
    model.eval()

    for nterms in range(trained_nterms+1, 6):
        print(f'Now testing generalization of model {model_name} to nterms = {nterms}...')

        if term_types == 12:
            test_data = mcts_main.load_data(nterms, train_data=False)
        elif term_types == 24:
            test_data = mcts_interaction_main.load_data(nterms, train_data=False)
        else:
            raise Exception('Invalid number of terms', term_types)
        
        test_sequence_list = [d[0] for d in test_data]

        rmse_list = []
        best_state_list = []
        perfect_counts = 0

        for seq in test_sequence_list:
            if term_types == 12:
                rmse, best_state = mcts_main.eval_nn(seq, model)
            elif term_types == 24:
                rmse, best_state = mcts_interaction_main.eval_nn(seq, model)
            else:
                raise Exception('Invalid number of terms', term_types)
            rmse_list.append(rmse)
            best_state_list.append(best_state)
            if rmse == 0:
                perfect_counts += 1
        
        avg_rmse = sum(rmse_list)/len(rmse_list)
        print('Mean RMSE on test data:', avg_rmse)
        print('Number of perfectly solved examples:', perfect_counts)
