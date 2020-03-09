#!/bin/bash
#
#SBATCH --partition=qsu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=baseline
#SBATCH --output=res2.txt
#
#SBATCH --time=12:00:00
#SBATCH --mem=64G

module load python/3.6.1
source /oak/stanford/groups/zihuai/fredlu/SeqModel/venv/bin/activate
ml load py-numpy/1.17.2_py36
cd /oak/stanford/groups/zihuai/fredlu/SeqModel
python evaluate_scores.py -p mpra_e116 -b 2