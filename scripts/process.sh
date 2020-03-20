#!/bin/bash
#
#SBATCH --partition=qsu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=extract
#SBATCH --output=out3.txt
#
#SBATCH --time=8:00:00
#SBATCH --mem=10G

module load python/3.6.1
source /oak/stanford/groups/zihuai/fredlu/SeqModel/venv/bin/activate
ml load py-numpy/1.17.2_py36
cd /oak/stanford/groups/zihuai/fredlu/SeqModel/knockoffs
python pull_features.py -b controls_hg19 -rb -e
