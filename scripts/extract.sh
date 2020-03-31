#!/bin/bash
#
#SBATCH --partition=qsu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=extract
#SBATCH --output=out5.txt
#
#SBATCH --time=8:00:00
#SBATCH --mem=40G

module load python/3.6.1
source /oak/stanford/groups/zihuai/fredlu/SeqModel/venv/bin/activate
ml load py-numpy/1.17.2_py36
cd /oak/stanford/groups/zihuai/fredlu/SeqModel/knockoffs
python extract_knockoff_regions.py
