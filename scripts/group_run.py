from subprocess import check_call
import os

commands = [
    'python pull_features.py -b top_W_matched_hg19_100 -g -r -rb -e -s mean -m',
    'python pull_features.py -b top_P_matched_hg19_100 -g -r -rb -e -s mean -m',
    'python pull_features.py -b top_W_matched_hg19_200 -g -r -rb -e -s mean -m',
    'python pull_features.py -b top_P_matched_hg19_200 -g -r -rb -e -s mean -m',
    'python pull_features.py -b top_W_matched_hg19_300 -g -r -rb -e -s mean -m',
    'python pull_features.py -b top_P_matched_hg19_300 -g -r -rb -e -s mean -m',
    'python pull_features.py -b top_W_matched_hg19_500 -g -r -rb -e -s mean -m',
    'python pull_features.py -b top_P_matched_hg19_500 -g -r -rb -e -s mean -m',
    'python pull_features.py -b top_W_matched_hg19_1000 -g -r -rb -e -s mean -m',
    'python pull_features.py -b top_P_matched_hg19_1000 -g -r -rb -e -s mean -m',
    'python pull_features.py -b AD_sig -g -r -rb -e -s mean -sm'
]


# Output the commands to a sbatch script then call sbatch.
with open("./process.sh") as f:
    template = f.read()

for name, command in enumerate(commands):
    sbatch = template.replace("COMMAND", command)
    sbatch = sbatch.replace("NAME", str(name))

    with open("spawn_tmp.sh", 'w') as f:
        f.write(sbatch)

    print(command)
    check_call("sbatch spawn_tmp.sh", shell=True)

os.remove("spawn_tmp.sh")
