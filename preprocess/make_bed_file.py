import argparse
from utils.bed_utils import get_bed_from_mpra, load_bed_file
from constants import PROCESSED_DIR


PROJ_CHOICES = ['mpra_e116']


def main():

    bedfile = load_bed_file('mpra_e116')


    bed_train, bed_test = split_train_test(bedfile, test_frac=0.2, seed=9999)

    # merge bed_train/test with mpra dataset for roadmap

    # same for eigen

    # same for regBase


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', choices=PROJ_CHOICES)
    args = parser.parse_args()

    bedfile = get_bed_from_mpra('e116')

