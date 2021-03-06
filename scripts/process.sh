#!/bin/bash
#
#SBATCH --partition=qsu,owners
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name="NAME"
#SBATCH --output=OUTFILE.txt
#
#SBATCH --time=8:00:00
#SBATCH --mem=8G

module load python/3.6.1
source /oak/stanford/groups/zihuai/fredlu/SeqModel/venv/bin/activate
ml load py-numpy/1.17.2_py36
cd /oak/stanford/groups/zihuai/fredlu/SeqModel/knockoffs

COMMAND

# done
echo "Done"