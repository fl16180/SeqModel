import numpy as np

from constants import *
from datasets import *
from utils.model_utils import load_model
from utils.data_utils import get_roadmap_col_order


class Evaluator:
    def __init__(self, trained_data='mpra_e116',
                 eval_data='mpra_nova'):
        self.trained_data = trained_data
        self.eval_data = eval_data
        self.ref = None
        self.model = None
        self.split = None
        self.X = None
        self.y = None
        self.scores = None
    
    def setup_data(self, model='standard', split='all'):
        self.model = model
        self.split = split

        if model in ['glm', 'standard']:
            df = load_data_set(self.eval_data, split=split, make_new=False)
            roadmap_cols = get_roadmap_col_order(order='marker')
            df[roadmap_cols] = np.log(df[roadmap_cols])

            proc = Processor(self.trained_data)
            proc.load(model)
            df = proc.transform(df)

            self.ref = df[['chr', 'pos', 'Label']].copy()
            X = df.drop(['chr', 'pos', 'Label'], axis=1) \
                .values \
                .astype(np.float32)
            y = df['Label'].values

        elif model == 'neighbors':
            X_neighbor = load_neighbors_set(self.eval_data,
                                            split=split,
                                            n_neigh=N_NEIGH,
                                            sample_res=SAMPLE_RES)
            X_neighbor = np.log(X_neighbor.astype(np.float32))

            df = load_data_set(self.eval_data, split=split)
            roadmap_cols = get_roadmap_col_order(order='marker')
            df[roadmap_cols] = np.log(df[roadmap_cols])

            proc = Processor(self.trained_data)
            proc.load(model)
            df = proc.transform(df)

            rm_cols = [f'{x}-E116' for x in ROADMAP_MARKERS]
            # rm_cols = get_roadmap_col_order(order='marker')
            X_score = df.drop(['chr', 'pos', 'Label'] + rm_cols, axis=1) \
                        .values \
                        .astype(np.float32)
            y = df['Label'].values
            assert X_neighbor.shape[0] == y.shape[0]

            X_neighbor = X_neighbor.reshape(
                X_neighbor.shape[0], X_neighbor.shape[1] * X_neighbor.shape[2])
            X = np.hstack((X_score, X_neighbor))
        self.X = X
        self.y = y

    def predict_model(self):
        net = load_model(self.trained_data, self.model)
        scores = net.predict_proba(self.X)
        self.scores = scores[:, 1]

    def save_scores(self):
        proj_dir = PROCESSED_DIR / self.eval_data

        try:
            df = pd.read_csv(proj_dir / 'output' / f'nn_preds_{self.eval_data}.csv',
                            sep=',')
        except FileNotFoundError:
            df = pd.read_csv(proj_dir / f'matrix_{self.split}.csv', sep=',')
            cols = ['chr', 'pos', 'Label', f'NN_{self.model}']
            df = df.loc[:, cols]
        
        assert np.all(self.y == df.Label)
        
        df[f'NN_{self.model}'] = self.scores
        df.to_csv(proj_dir / 'output' / f'nn_preds_{self.eval_data}.csv',
                sep=',', index=False)
