#!/bin/bash
#
#SBATCH --partition=qsu,owners
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --job-name=script3
#SBATCH --output=result4.txt
#
#SBATCH --time=10:00:00
#SBATCH --mem=24G

module load python/3.6.1
source /oak/stanford/groups/zihuai/fredlu/SeqModel/venv/bin/activate
ml load py-numpy/1.17.2_py36
cd /oak/stanford/groups/zihuai/fredlu/SeqModel/knockoffs
python fit_models.py -d all_mean_top_W_matched_hg19_500 -f
