import os

for nterms in [2, 3]:
    for dimension in [256, 512]:
        for layers in [5, 6]:
            for outer_iters in [5, 6, 7]:
                for model_type in ['MLP', 'GRU', 'TFR']:

                    s =f"""#!/bin/bash
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yl708@duke.edu     # Where to send mail:
#SBATCH --partition=scavenger-gpu
#SBATCH --exclusive
#SBATCH --time='7-0'
#SBATCH --chdir='/work/yl708/CS-590-Project'
#SBATCH --mem=8G

source ~/.bashrc
source ~/.bash_profile
cd /work/yl708/CS-590-Project
date
hostname

conda activate torch
"""

                    s += f'echo "mcts_interaction_main.py --nterms {nterms} --model_type {model_type} --dim {dimension} --layers {layers} --outer_iters {outer_iters}"\n'
                    s += f'python mcts_interaction_main.py --nterms {nterms} --model_type {model_type} --dim {dimension} --layers {layers} --outer_iters {outer_iters}\n'

                    filename = f'mcts_{nterms}_{model_type}_{dimension}_{layers}_{outer_iters}.sh'

                    with open(filename, 'w') as script:
                        script.write(s)

                    os.system(f'sbatch {filename}')
                    os.system(f'rm {filename}')


