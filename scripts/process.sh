#!/bin/bash
#
#SBATCH --partition=qsu
#SBATCH --nodes=1
#SBATCH --job-name=preprocess
#SBATCH --output=res.txt
#
#SBATCH --ntasks=1
#SBATCH --time=07:00:00
#SBATCH --mem=4G

module load python/3.6.1
source /oak/stanford/groups/zihuai/fredlu/SeqModel/venv/bin/activate
cd /oak/stanford/groups/zihuai/fredlu/SeqModel
python setup_project_data.py -p mpra_nova -rb -e --split train-test
