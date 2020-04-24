#!/bin/bash
#
#SBATCH --partition=qsu,owners
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --job-name=search
#SBATCH --output=hparam_fc_mean_top_W_1000.txt
#
#SBATCH --time=14:00:00
#SBATCH --mem=24G

module load python/3.6.1
source /oak/stanford/groups/zihuai/fredlu/SeqModel/venv/bin/activate
ml load py-numpy/1.17.2_py36
cd /oak/stanford/groups/zihuai/fredlu/SeqModel/knockoffs
python search_hparam.py -d all_mean_top_W_matched_hg19_1000.tsv -i 500
