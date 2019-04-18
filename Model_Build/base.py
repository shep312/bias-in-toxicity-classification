import gc
import yaml
from numpy import mean, nan
from pathlib import Path
from datetime import datetime
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from .utils import *

class BaseClassifier:
    """
    Base class of Keras classifiers
    """

    def __init__(self, params):
        """ Set generic parameters """
        self.run_config = params
        self.embedding_path = params['embedding_path']
        self.identity_data_path = params['train_data_path']
        self.batch_size = params['batch_size']
        self.epochs = params['max_epochs']
        self.bias_identities = get_identities()
        self.identity_weight = params['identity_weight']
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H.%M.%S')
        self.comp_metric = None
        self.cv_comp_metrics = []
        self.result = {}
        self.cv_results = []
        self.model = None

    def get_n_unique_words(self):
        data = pd.read_csv(self.run_config['sequence_path'],
                           nrows=self.run_config['debug_size'])
        return len(pd.unique(data.values.ravel('K')))

    def load_identity_data(self):
        identity_data = pd.read_csv(self.identity_data_path,
                                    usecols=self.bias_identities,
                                    nrows=self.run_config['debug_size'])
        return identity_data.fillna(0).astype(bool)

    def embedding_as_keras_layer(self):
        """ Load specified embedding as a Keras layer """
        if self.embedding_path:
            embedding_matrix = pd.read_csv(self.embedding_path)
            return Embedding(embedding_matrix.shape[0],
                             embedding_matrix.shape[1],
                             weights=[embedding_matrix],
                             trainable=False)
        else:
            return Embedding(self.get_n_unique_words(), 100)

    def create_model(self):
        pass

    def train(self, X, y, train_idx=None, val_idx=None):
        """ Define a training function with early stopping """
        if train_idx is not None:
            X_train = X[train_idx]
            X_val = X[val_idx]
            if isinstance(y, pd.Series):
                y_train = y.iloc[train_idx]
                y_val = y.iloc[val_idx]
            else:
                y_train = y[train_idx]
                y_val = y[val_idx]
            validation_data = [X_val, y_val]
        else:
            X_train = X
            y_train = y
            validation_data = None
        del X, y
        gc.collect()

        self.model = self.create_model()
        sample_weights = np.ones([X_train.shape[0], ])
        if self.identity_weight != 1:
            identity_data = self.load_identity_data()
            if train_idx is not None:
                identity_data = identity_data.iloc[train_idx, :]
            identity_mask = identity_data.any(axis=1)
            sample_weights[identity_mask] = sample_weights[identity_mask] \
                * self.identity_weight
            del identity_data
            gc.collect()

        early_stopping_loss = 'loss' if train_idx is None else 'val_loss'

        self.result = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            sample_weight=sample_weights,
            callbacks=[
                EarlyStopping(
                    monitor=early_stopping_loss,
                    min_delta=0.001,
                    patience=3,
                    verbose=1
                )
            ]
        )

        if train_idx is not None:
            print('Computing bias metrics...')
            identity_data = self.load_identity_data()
            identity_data = identity_data.iloc[val_idx, :]

            y_pred = self.model.predict(X_val)
            bias_metrics_df = compute_bias_metrics_for_model(
                identity_data, self.bias_identities, y_val, y_pred
            )
            self.comp_metric = get_final_metric(
                bias_metrics_df, roc_auc_score(y_val, y_pred)
            )
            print('Bias metrics computed. Score = {:.4f}'
                  .format(self.comp_metric))

    def cv(self, X, y, cv=StratifiedKFold(3)):
        """ Apply training function in CV fold """
        for fold_no, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print('Fitting fold {} / {}\n'.format(fold_no + 1, cv.get_n_splits()))

            self.train(X, y, train_idx, val_idx)
            self.cv_comp_metrics.append(self.comp_metric)
            self.cv_results.append(self.result.history)

            K.clear_session()
            del self.model
            gc.collect()

        self.run_config['cv_comp_metrics'] = self.cv_comp_metrics
        self.run_config['cv_results'] = self.cv_results

    def save(self, path):
        if len(self.cv_comp_metrics) > 0:
            score = mean(self.cv_comp_metrics)
        else:
            score = self.comp_metric
        score = nan if score is None else score

        out_dir = Path(path)
        out_dir = out_dir / '{}_score_{:.4f}'.format(self.run_timestamp, score)
        out_dir.mkdir(parents=True, exist_ok=True)

        results_out_path = out_dir / 'CONFIG_{}.yaml'.format(self.__name__)
        self.run_config['cv_comp_metrics'] = \
            [str(x) for x in self.run_config['cv_comp_metrics']]
        self.run_config['cv_results'] = \
            {key: [str(x) for x in val]
             for key, val in self.run_config['cv_results'][0].items()}
        with results_out_path.open('w') as f:
             yaml.dump(self.run_config, f)

        model_out_path = out_dir / 'MODEL_{}.h5'.format(self.__name__)
        self.model.save(str(model_out_path))
