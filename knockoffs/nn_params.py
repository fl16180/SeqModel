topW_100 = {
    'batch_size': 256,
    'optimizer__weight_decay': 1e-6,
    'lr': 2.8e-3,
    'max_epochs': 40,
    'module__n_units': (400, 300, 500),
    'module__dropout': 0.062
}

topW_200 = {
    'batch_size': 256,
    'optimizer__weight_decay': 1.9e-4,
    'lr': 2.1e-3,
    'max_epochs': 50,
    'module__n_units': (500, 400),
    'module__dropout': 0.038
}

topW_300 = {
    'batch_size': 256,
    'optimizer__weight_decay': 1.8e-4,
    'lr': 1.9e-3,
    'max_epochs': 50,
    'module__n_units': (500, 200),
    'module__dropout': 0.2
}

topW_500 = {
    'batch_size': 256,
    'optimizer__weight_decay': 1e-4,
    'lr': 1.2e-3,
    'max_epochs': 50,
    'module__n_units': (500, 500),
    'module__dropout': 0.087
}


topW_1000 = {
    'batch_size': 256,
    'optimizer__weight_decay': 3e-5,
    'lr': 2e-3,
    'max_epochs': 50,
    'module__n_units': (500, 600, 300),
    'module__dropout': 0.08
}

param_lookup = {
    'all_mean_top_W_matched_hg19_100': topW_100,
    'all_mean_top_W_matched_hg19_200': topW_200,
    'all_mean_top_W_matched_hg19_300': topW_300,
    'all_mean_top_W_matched_hg19_500': topW_500,
    'all_mean_top_W_matched_hg19_1000': topW_1000,
    'all_mean_top_P_matched_hg19_100': topW_100,
    'all_mean_top_P_matched_hg19_200': topW_200,
    'all_mean_top_P_matched_hg19_300': topW_300,
    'all_mean_top_P_matched_hg19_500': topW_500,
    'all_mean_top_P_matched_hg19_1000': topW_1000
}