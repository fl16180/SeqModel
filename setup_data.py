
def build(dataset):

    os.makedirs(cfg.TRAIN_DIR, exist_ok=True)
    os.makedirs(cfg.TEST_DIR, exist_ok=True)
    os.makedirs(cfg.BENCH_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    data, bench = load_labeled_data(GenomeDatasets[dataset])
    bench.to_csv(os.path.join(cfg.BENCH_DIR, '{0}_benchmarks.csv'.format(dataset)), index=False)

    train, test = split_train_test(data, test_frac=0.15, seed=100)
    meta_train = train[['chr', 'pos', 'rs', 'Label']]
    X_train = train.drop(['chr', 'pos', 'rs', 'Label'], axis=1)
    X_train = rearrange_by_epigenetic_marker(X_train)
    train = pd.concat([meta_train, X_train], axis=1)

    meta_test = test[['chr', 'pos', 'rs', 'Label']]
    X_test = test.drop(['chr', 'pos', 'rs', 'Label'], axis=1)
    X_test = rearrange_by_epigenetic_marker(X_test)
    test = pd.concat([meta_test, X_test], axis=1)

    train.to_csv(os.path.join(cfg.TRAIN_DIR, '{0}_train.csv'.format(dataset)), index=False)
    test.to_csv(os.path.join(cfg.TEST_DIR, '{0}_test.csv'.format(dataset)), index=False)

    val_train, val_test = add_valley_scores(dataset)
    val_train.to_csv(os.path.join(cfg.TRAIN_DIR, '{0}_valley_train.csv'.format(dataset)), index=False)
    val_test.to_csv(os.path.join(cfg.TEST_DIR, '{0}_valley_test.csv'.format(dataset)), index=False)

    train_seq, test_seq = merge_seq_with_mpra(dataset)
    train_seq.to_csv(os.path.join(cfg.TRAIN_DIR, '{0}_seq_train.csv'.format(dataset)), index=False)
    test_seq.to_csv(os.path.join(cfg.TEST_DIR, '{0}_seq_test.csv'.format(dataset)), index=False)




if __name__ == '__main__':
    build('E116')
    # build('E118')
    # build('E123')
