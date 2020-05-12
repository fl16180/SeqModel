#!/bin/bash
#
#SBATCH --partition=qsu,owners
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --job-name=extract
#SBATCH --output=out.txt
#
#SBATCH --time=14:00:00
#SBATCH --mem=15G

module load python/3.6.1
source /oak/stanford/groups/zihuai/fredlu/SeqModel/venv/bin/activate
ml load py-numpy/1.17.2_py36
cd /oak/stanford/groups/zihuai/fredlu/SeqModel/
python setup_neighbor_data.py -p mpra_e116 -e -t E116 -s all -npr 40,25
