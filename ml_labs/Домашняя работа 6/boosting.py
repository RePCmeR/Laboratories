from __future__ import annotations

from collections import defaultdict
from typing import Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score

from tqdm.auto import tqdm

from sklearn.base import ClassifierMixin


class BoostingClassifier(ClassifierMixin):

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: dict | None = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = True,
        eval_metric: str | None = None,
        early_stopping_rounds: int = 0,
        cat_features: Iterable[int] | None = None,
        subsample: float = 1.0,
        bagging_temperature: float = 1.0,
        bootstrap_type: str | None = 'Bernoulli',
        rsm: float = 1.0,
        goss: bool = False,
        goss_k: float = 0.2,
        loss: str = 'BCE',
        focal_gamma: float = 2.0
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = [0] * (n_estimators)
        self.gammas = [0] * (n_estimators)

        self.random_state = random_state  # не забудьте вставить его везде, где у вас возникает рандом
        self.verbose = verbose

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        if loss == 'FocalLoss':
            self.loss_fn = lambda y, z: - ((1 - self.sigmoid(y * z)) ** focal_gamma * np.log(self.sigmoid(y * z))).mean()
            self.grad_fn = lambda y, z: y * (1 - self.sigmoid(y * z)) ** focal_gamma * ((1 - self.sigmoid(y * z)) - focal_gamma * self.sigmoid(y * z) * np.log(self.sigmoid(y * z)))
        else:
            self.loss_fn = lambda y, z: -np.log(np.clip(self.sigmoid(y * z))).mean()
            self.grad_fn = lambda y, z: y * (1 - self.sigmoid(y * z))

        self._n_fitted = 0
        self._train_predictions = None

        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

        if cat_features is not None:
            self.cat_features = list(cat_features)
        else:
            self.cat_features = None

        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type

        self.rsm = rsm
        self._feature_masks = [None] * n_estimators

        self.goss = goss
        self.goss_k = goss_k

    def _cat_fit(self, X: np.ndarray, y: np.ndarray):
        pos = (y == 1).astype(float)
        prior = pos.mean()

        sort = np.argsort(X[:, 2], kind='mergesort')  # сортим по году
        X_sorted = X[sort]
        pos_sorted = pos[sort]

        self._cat_mapping = {}
        self._cat_defaults = {}
        online_values = np.zeros((X.shape[0], len(self.cat_features)))

        for i in range(len(self.cat_features)):
            col = self.cat_features[i]
            col_data = X_sorted[:, col]

            self._cat_mapping[col] = {v: pos[X[:, col] == v].mean() for v in np.unique(X[:, col])}
            self._cat_defaults[col] = prior

            cum_sum = defaultdict(float)
            cum_cnt = defaultdict(int)
            for j in range(X.shape[0]):
                cat_val = col_data[j]
                if cum_cnt[cat_val] > 0:
                    online_values[j, i] = cum_sum[cat_val] / cum_cnt[cat_val]
                else:
                    online_values[j, i] = prior
                cum_sum[cat_val] += pos_sorted[j]
                cum_cnt[cat_val] += 1

        self._cat_online_values = online_values[np.argsort(sort)]

    def _cat_transform(self, X: np.ndarray):
        if (hasattr(self, '_cat_online_values') and self._cat_online_values.shape[0] == X.shape[0]):
            X_new = X.copy()
            for i in range(len(self.cat_features)):
                col = self.cat_features[i]
                X_new[:, col] = self._cat_online_values[:, i]
            return X_new.astype(float)
        elif hasattr(self, '_cat_mapping') and self._cat_mapping is not None:
            X_new = X.copy()
            for i in range(len(self.cat_features)):
                col = self.cat_features[i]
                mapping = self._cat_mapping[col]
                default = self._cat_defaults[col]
                X_new[:, col] = [mapping.get(v, default) for v in X[:, col]]
            return X_new.astype(float)
        else:
            return X

    def _get_bootstrap_sample(self, n):
        if self.bootstrap_type is None:
            return np.ones(n, dtype=bool), None

        rng = np.random.RandomState(self.random_state + self._n_fitted)
        if self.bootstrap_type == 'Bernoulli':
            mask = rng.random(n) < self.subsample
            if mask.sum() == 0:
                mask = np.ones(n, dtype=bool)
            return mask, None
        else:
            U = rng.random(n)
            w = (-np.log(U)) ** self.bagging_temperature
            return slice(None), w

    def _get_feature_mask(self, n_features):
        rng = np.random.RandomState(self.random_state + self._n_fitted)
        mask = rng.random(n_features) < self.rsm
        if mask.sum() == 0:
            mask[np.random.randint(0, n_features)] = True
        return mask

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        grad = self.grad_fn(y, self._train_predictions)

        feature_mask = self._get_feature_mask(X.shape[1])

        if self.goss:
            top = int(X.shape[0] * self.goss_k)

            sorted = np.argsort(-np.abs(grad))
            top_idx = sorted[:top]
            bottom_idx = sorted[top:]

            rng = np.random.RandomState(self.random_state + self._n_fitted)
            selected = bottom_idx[rng.random(len(bottom_idx)) < self.subsample]

            all_idx = np.concatenate([top_idx, selected])
            X_fit = X[all_idx]
            grad_fit = grad[all_idx]

            w_small = (1.0 - self.goss_k) / self.subsample if self.subsample > 0 else 1.0
            weights = np.concatenate([np.ones(top), np.full(len(selected), w_small)])
            sw = weights
        else:
            mask, sample_weight = self._get_bootstrap_sample(X.shape[0])
            if self.bootstrap_type == 'Bernoulli':
                X_fit = X[mask]
                grad_fit = grad[mask]
                sw = None
            elif self.bootstrap_type == 'Bayesian':
                X_fit = X
                grad_fit = grad
                sw = sample_weight
            else:
                X_fit = X
                grad_fit = grad
                sw = None

        if feature_mask is not None:
            X_fit = X_fit[:, feature_mask]
            X_pred = X[:, feature_mask]
        else:
            X_pred = X

        params = self.base_model_params.copy()
        params.setdefault('random_state', self.random_state)
        model = self.base_model_class(**params)
        model.fit(X_fit, grad_fit, sample_weight=sw)
        pred = model.predict(X_pred)
        gamma = self._find_optimal_gamma(y, self._train_predictions, pred)

        self.models[self._n_fitted] = model
        self.gammas[self._n_fitted] = gamma
        self._feature_masks[self._n_fitted] = feature_mask

        self._train_predictions += self.learning_rate * gamma * pred
        self._n_fitted += 1

        self.history['train_loss'].append(self.loss_fn(y, self._train_predictions))
        roc_auc = roc_auc_score(y == 1, self.predict_proba(X)[:, 1])
        self.history['train_roc_auc'].append(roc_auc)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, eval_set: Tuple[np.ndarray, np.ndarray] | None = None, use_best_model: bool = False) -> None:

        self._train_predictions = np.zeros(X_train.shape[0])
        self.classes_ = np.unique(y_train)  # не рекомендуется убирать, нужно для калибровки
        self._n_fitted = 0

        if self.cat_features is not None:
            self._cat_fit(X_train, y_train)
            X_train = self._cat_transform(X_train)

        if eval_set is not None:
            X_val, y_val = eval_set
            if self.cat_features is not None:
                X_val = self._cat_transform(X_val)

        best_score = None
        best_iteration = 0
        counter = 0
        stop_flag = False

        estimator_range = range(self.n_estimators)
        if self.verbose:
            estimator_range = tqdm(estimator_range)

        for _ in estimator_range:
            self.partial_fit(X_train, y_train)

            if eval_set is not None:
                scores_temp = np.zeros(X_val.shape[0])
                for i in range(self._n_fitted):
                    fm = self._feature_masks[i]
                    X_val_masked = X_val[:, fm] if fm is not None else X_val
                    scores_temp += self.learning_rate * self.gammas[i] * self.models[i].predict(X_val_masked)

                self.history['valid_loss'].append(self.loss_fn(y_val, scores_temp))
                roc_auc_temp = roc_auc_score(y_val == 1, self.sigmoid(scores_temp))
                self.history['valid_roc_auc'].append(roc_auc_temp)

                if self.eval_metric is not None and self.early_stopping_rounds > 0:
                    current_score = self.history[self.eval_metric][-1]

                    if best_score is None:
                        best_score = current_score
                        best_iteration = self._n_fitted - 1
                        counter = 0
                    elif 'loss' in self.eval_metric:
                        if current_score < best_score:
                            best_score = current_score
                            best_iteration = self._n_fitted - 1
                            counter = 0
                        else:
                            counter += 1
                            if counter >= self.early_stopping_rounds:
                                stop_flag = True
                                break
                    else:
                        if current_score > best_score:
                            best_score = current_score
                            best_iteration = self._n_fitted - 1
                            counter = 0
                        else:
                            counter += 1
                            if counter >= self.early_stopping_rounds:
                                stop_flag = True
                                break

        if use_best_model and stop_flag:
            self.models = self.models[:best_iteration + 1]
            self.gammas = self.gammas[:best_iteration + 1]
            self._feature_masks = self._feature_masks[:best_iteration + 1]
            self._n_fitted = best_iteration + 1
            self._train_predictions = None

        # чтобы было удобнее смотреть
        for key in self.history:
            self.history[key] = np.array(self.history[key])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._cat_transform(X)
        scores = np.zeros(X.shape[0])
        for i in range(self._n_fitted):
            fm = self._feature_masks[i]
            X_masked = X[:, fm] if fm is not None else X
            scores += self.learning_rate * self.gammas[i] * self.models[i].predict(X_masked)
        proba = self.sigmoid(scores)
        return np.column_stack([1 - proba, proba])

    def _find_optimal_gamma(
        self,
        y: np.ndarray,
        old_predictions: np.ndarray,
        new_predictions: np.ndarray
    ) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def get_feature_importance(self, X: np.ndarray | None = None, y: np.ndarray | None = None, type = "split"):
        if self._feature_masks[0] is not None:
            n = len(self._feature_masks[0])
        elif self._n_fitted > 0:
            n = self.models[0].n_features_in_
        else:
            n = 0

        total = np.zeros(n)

        if type == 'split':
            for i in range(self._n_fitted):
                model = self.models[i]
                mask = self._feature_masks[i]
                imp = model.feature_importances_
                if mask is not None:
                    full = np.zeros(n)
                    full[mask] = imp
                else:
                    full = imp
                total += full * self.gammas[i]
        elif type == 'gain':
            for i in range(self._n_fitted):
                model = self.models[i]
                mask = self._feature_masks[i]
                tree = model.tree_

                if mask is not None:
                    map = np.where(mask)[0]
                else:
                    map = np.arange(n)

                for node in range(tree.node_count):
                    if tree.children_left[node] == tree.children_right[node]:
                        continue
                    n_left = tree.n_node_samples[tree.children_left[node]]
                    imp_left = tree.impurity[tree.children_left[node]]
                    n_right = tree.n_node_samples[tree.children_right[node]]
                    imp_right = tree.impurity[tree.children_right[node]]
                    reduction = tree.n_node_samples[node] * tree.impurity[node] - n_left * imp_left - n_right * imp_right
                    feature = map[tree.feature[node]]
                    total[feature] += reduction * self.gammas[i]

        if total.sum() > 0:
            total /= total.sum()
        return total

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])

    def plot_history(self, keys: str | Iterable[str]) -> None:
        if isinstance(keys, str):
            keys = [keys]
        plt.figure(figsize=(12, 7))
        for key in keys:
            if key in self.history:
                plt.plot(self.history[key], label=key)
        plt.title('История')
        plt.xlabel('Итерация')
        plt.ylabel('Значение')
        plt.legend()
        plt.grid(True)
        plt.show()