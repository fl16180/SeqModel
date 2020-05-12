import argparse
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score

from constants import *
from datasets import *
from models import *
from utils.model_utils import *
from utils.data_utils import get_roadmap_col_order, load_mpra_data
from utils.metrics import show_prec_recall, pr_summary


def concat_addl_scores(project, na_thresh=0.05):
    """ output score file contains chr, pos, label, nn_scores. For evaluation
    we want additional scores for regbase, eigen, etc. Concat these to the
    score file, excluding non-E116 roadmap scores
    """
    proj_dir = PROCESSED_DIR / project

    addl = pd.read_csv(proj_dir / 'matrix_all.csv', sep=',')
    addl.drop(['chr', 'pos', 'Label'], axis=1, inplace=True)

    omit_roadmap = [x for x in get_roadmap_col_order() if x[-3:] != '116']
    addl.drop(omit_roadmap, axis=1, inplace=True)

    # drop scores that have >5% NaNs from metrics (were dropped from nn as well)
    na_filt = (addl.isna().sum() > na_thresh * len(addl))
    omit_cols = addl.columns[na_filt].tolist()
    omit_cols += [x + '_PHRED' for x in omit_cols if x + '_PHRED' in addl.columns]
    addl.drop(omit_cols, axis=1, inplace=True)

    scores = pd.read_csv(proj_dir / 'output' / f'nn_preds_{project}.csv',
                         sep=',')
    scores = pd.concat([scores, addl], axis=1)
    return scores


def add_mpra_benchmarks(args, scores, benchmarks=['GNET']):
    _, bench = load_mpra_data(args.project, benchmarks=True)
    bench = bench.loc[:, ['chr', 'pos'] + benchmarks]
    scores = pd.merge(scores, bench, on=['chr', 'pos'])
    return scores


def merge_with_validation_info(scores, eval_proj):
    ext_val_path = MPRA_DIR / MPRA_TABLE[eval_proj][0]
    dat = pd.read_csv(ext_val_path, sep='\t')
    dat.rename(columns={'chrom': 'chr', 'Hit': 'Label'}, inplace=True)
    dat = dat.loc[:,
        ['chr', 'pos', 'Pool', 'pvalue_expr',
        'padj_expr', 'pvalue_allele', 'padj_allele', 'Label']
    ]
    dat['chr'] = dat['chr'].map(lambda x: x[3:])
    dat[['chr', 'pos']] = dat[['chr', 'pos']].astype(int)

    scores.drop('Label', axis=1, inplace=True)
    scores = pd.merge(dat, scores, on=['chr', 'pos'])
    return scores


def score_metric_comparison(scores, metric='AUC'):

    # def make_scorer(score_fn):
    #     def scorer(x):
    #         try:
    #             return score_fn(scores.Label[~x.isna()], x[~x.isna()])
    #         except ValueError:
    #             return np.nan
    #     return scorer

    if metric == 'AUC':
        # scorer = lambda x: make_scorer(roc_auc_score)
        scorer = lambda x: roc_auc_score(scores.Label[~x.isna()], x[~x.isna()])
    elif metric == 'APR':
        scorer = lambda x: average_precision_score(scores.Label[~x.isna()], x[~x.isna()])

    try:
        results = scores.drop(
            ['chr', 'pos', 'Label', 'Pool', 'pvalue_expr',
            'padj_expr', 'pvalue_allele', 'padj_allele'], axis=1).apply(scorer)
    except KeyError:
        results = scores.drop(['chr', 'pos', 'Label'], axis=1).apply(scorer)
    # print(results.sort_values(ascending=False))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', default='mpra_e116')
    parser.add_argument('--eval_proj', '-e', default='mpra_nova')
    args = parser.parse_args()

    out_dir = PROCESSED_DIR / args.project / 'output'
    eval_out_dir = PROCESSED_DIR / args.eval_proj / 'output'

    # # evaluate saved models on mpra_nova data and save scores to file
    # for mod in ['standard', 'neighbors']:
    #     evl = Evaluator(trained_data=args.project, eval_data=args.eval_proj)
    #     evl.setup_data(model=mod, split='all')
    #     evl.predict_model()
    #     evl.save_scores()

    # add scores and compute internal CV metrics
    scores = concat_addl_scores(args.project)
    scores = add_mpra_benchmarks(args, scores, ['GNET'])
    scores.index.names = ['scores']
    scores.reset_index() \
          .to_csv(out_dir / f'all_scores_nn_preds_{args.project}.csv',
                  index=False)

    int_scores = score_metric_comparison(scores, 'AUC')
    int_scores = int_scores.to_frame(name='cv_e116')

    # add scores and compute external validation metrics
    nova_scores = concat_addl_scores(args.eval_proj)
    n_unique_var = nova_scores.shape[0]

    nova_scores = merge_with_validation_info(nova_scores, args.eval_proj)
    nova_scores.index.names = ['scores']
    nova_scores.reset_index() \
               .to_csv(eval_out_dir / f'all_scores_nn_preds_{args.eval_proj}.csv',
                       index=False)


    # --- compute detailed AUC comparisons --- #
    nova_scores = nova_scores.copy()
    n_var_table = {}
    bonf_01, bonf_05 = 0.01 / n_unique_var, 0.05 / n_unique_var
    Hit_col = nova_scores['Label'].copy()       # default p-adj thresholding in data

    # # Bonferroni 0.01
    # nova_scores['Label'] = (nova_scores['pvalue_expr'] < bonf_01).astype(int)
    # nova_scores_clean = nova_scores.drop_duplicates(subset=['chr', 'pos', 'Label'])
    # ext_scores_expr_b01 = score_metric_comparison(nova_scores_clean, 'AUC')
    # ext_scores_expr_b01 = ext_scores_expr_b01.to_frame(name='nova_expr_bonf_0.01')
    # n_var_table['expr_bonf_0.01'] = f"{nova_scores_clean['Label'].sum()}/{len(nova_scores_clean)}"

    # nova_scores['Label'] = (nova_scores['pvalue_allele'] < bonf_01).astype(int)
    # nova_scores_clean = nova_scores.drop_duplicates(subset=['chr', 'pos', 'Label'])
    # ext_scores_alle_b01 = score_metric_comparison(nova_scores_clean, 'AUC')
    # ext_scores_alle_b01 = ext_scores_alle_b01.to_frame(name='nova_alle_bonf_0.01')
    # n_var_table['alle_bonf_0.01'] = f"{nova_scores_clean['Label'].sum()}/{len(nova_scores_clean)}"

    # # Bonferroni 0.05
    # nova_scores['Label'] = (nova_scores['pvalue_expr'] < bonf_05).astype(int)
    # nova_scores_clean = nova_scores.drop_duplicates(subset=['chr', 'pos', 'Label'])
    # ext_scores_expr_b05 = score_metric_comparison(nova_scores_clean, 'AUC')
    # ext_scores_expr_b05 = ext_scores_expr_b05.to_frame(name='nova_expr_bonf_0.05')
    # n_var_table['expr_bonf_0.05'] = f"{nova_scores_clean['Label'].sum()}/{len(nova_scores_clean)}"

    # nova_scores['Label'] = (nova_scores['pvalue_allele'] < bonf_05).astype(int)
    # nova_scores_clean = nova_scores.drop_duplicates(subset=['chr', 'pos', 'Label'])
    # ext_scores_alle_b05 = score_metric_comparison(nova_scores_clean, 'AUC')
    # ext_scores_alle_b05 = ext_scores_alle_b05.to_frame(name='nova_alle_bonf_0.05')
    # n_var_table['alle_bonf_0.05'] = f"{nova_scores_clean['Label'].sum()}/{len(nova_scores_clean)}"
    
    # # Adjusted p-values
    # nova_scores['Label'] = Hit_col.isin(['Expr', 'Both']).astype(int)
    # nova_scores_clean = nova_scores.drop_duplicates(subset=['chr', 'pos', 'Label'])
    # ext_scores_expr_adj = score_metric_comparison(nova_scores_clean, 'AUC')
    # ext_scores_expr_adj = ext_scores_expr_adj.to_frame(name='nova_expr_adj')
    # n_var_table['expr_adj'] = f"{nova_scores_clean['Label'].sum()}/{len(nova_scores_clean)}"

    # nova_scores['Label'] = Hit_col.isin(['Allele', 'Both']).astype(int)
    # nova_scores_clean = nova_scores.drop_duplicates(subset=['chr', 'pos', 'Label'])
    # ext_scores_alle_adj = score_metric_comparison(nova_scores_clean, 'AUC')
    # ext_scores_alle_adj = ext_scores_alle_adj.to_frame(name='nova_alle_adj')
    # n_var_table['alle_adj'] = f"{nova_scores_clean['Label'].sum()}/{len(nova_scores_clean)}"

    # from functools import reduce
    # tab = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True),
    #              [int_scores,
    #               ext_scores_expr_b01, ext_scores_expr_b05, ext_scores_expr_adj,
    #               ext_scores_alle_b01, ext_scores_alle_b05, ext_scores_alle_adj]
    # )
    # tab.index.names = ['scores']
    # tab.reset_index().to_csv(out_dir / 'AUC_comparison.csv', index=False)

    # print(n_var_table)

    # --- prec-recall constraints --- #

    nova_scores['Label'] = (nova_scores['pvalue_expr'] < bonf_01).astype(int)
    nova_scores_clean = nova_scores.drop_duplicates(subset=['chr', 'pos', 'Label'])
    prdf = show_prec_recall(nova_scores_clean['Label'], nova_scores_clean['NN_standard'])
    prc = pr_summary(nova_scores_clean['Label'],
                     nova_scores_clean['NN_standard'],
                     precs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    print(prc)


    nova_scores['Label'] = (nova_scores['pvalue_expr'] < bonf_05).astype(int)
    nova_scores_clean = nova_scores.drop_duplicates(subset=['chr', 'pos', 'Label'])
    prdf = show_prec_recall(nova_scores_clean['Label'], nova_scores_clean['NN_standard'])
    prc = pr_summary(nova_scores_clean['Label'],
                     nova_scores_clean['NN_standard'],
                     precs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    print(prc)


    nova_scores['Label'] = Hit_col.isin(['Expr', 'Both']).astype(int)
    nova_scores_clean = nova_scores.drop_duplicates(subset=['chr', 'pos', 'Label'])
    prdf = show_prec_recall(nova_scores_clean['Label'], nova_scores_clean['NN_standard'])
    prc = pr_summary(nova_scores_clean['Label'],
                     nova_scores_clean['NN_standard'],
                     precs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    print(prc)

    nova_scores['Label'] = (nova_scores['pvalue_allele'] < bonf_01).astype(int)
    print(sum(nova_scores['Label']))
    nova_scores_clean = nova_scores.drop_duplicates(subset=['chr', 'pos', 'Label'])
    prdf = show_prec_recall(nova_scores_clean['Label'], nova_scores_clean['NN_standard'])
    prc = pr_summary(nova_scores_clean['Label'],
                     nova_scores_clean['NN_standard'],
                     precs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    print(prc)


    nova_scores['Label'] = (nova_scores['pvalue_allele'] < bonf_05).astype(int)
    print(sum(nova_scores['Label']))
    nova_scores_clean = nova_scores.drop_duplicates(subset=['chr', 'pos', 'Label'])
    prdf = show_prec_recall(nova_scores_clean['Label'], nova_scores_clean['NN_standard'])
    prc = pr_summary(nova_scores_clean['Label'],
                     nova_scores_clean['NN_standard'],
                     precs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    print(prc)


    nova_scores['Label'] = Hit_col.isin(['Allele', 'Both']).astype(int)
    print(sum(nova_scores['Label']))
    nova_scores_clean = nova_scores.drop_duplicates(subset=['chr', 'pos', 'Label'])
    prdf = show_prec_recall(nova_scores_clean['Label'], nova_scores_clean['NN_standard'])
    prc = pr_summary(nova_scores_clean['Label'],
                     nova_scores_clean['NN_standard'],
                     precs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    print(prc)

    # ext_scores_expr_b01 = score_metric_comparison(nova_scores_clean, 'AUC')
    # ext_scores_expr_b01 = ext_scores_expr_b01.to_frame(name='nova_expr_bonf_0.1')
    # n_var_table['expr_bonf_0.1'] = nova_scores['Label'].sum() 