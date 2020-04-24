from subprocess import check_call
import os


files = ['all_mean_top_W_matched_hg19_100.tsv',
         'all_mean_top_W_matched_hg19_200.tsv',
         'all_mean_top_W_matched_hg19_300.tsv',
         'all_mean_top_W_matched_hg19_500.tsv',
         'all_mean_top_W_matched_hg19_1000.tsv']

commands = [
    f'python search_hparam.py -d {x} -i 500' for x in files
]


# Output the commands to a sbatch script then call sbatch.
with open("./process.sh") as f:
    template = f.read()

for name, command in zip(files, commands):
    sbatch = template.replace("COMMAND", command)
    sbatch = sbatch.replace("NAME", str(name))
    sbatch = sbatch.replace("OUTFILE", str(name))

    with open("spawn_tmp.sh", 'w') as f:
        f.write(sbatch)

    print(command)
    check_call("sbatch spawn_tmp.sh", shell=True)

os.remove("spawn_tmp.sh")
