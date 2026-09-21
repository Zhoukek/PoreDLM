"""Held-out classification metrics and uncertainty grouped by physical read."""
import numpy as np
from scipy.sparse import coo_matrix


def classification_metrics(y_true, y_pred, top, classes, common_labels):
    classes = np.asarray(classes)
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    k = len(classes)
    if not len(y_true):
        raise ValueError('empty test/validation set')
    matrix = np.bincount(y_true * k + y_pred, minlength=k*k).reshape(k, k)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    correct = matrix.diagonal()
    supported = support > 0
    recall = np.divide(correct, support, out=np.zeros(k, dtype=float), where=supported)
    precision = np.divide(correct, predicted, out=np.zeros(k, dtype=float), where=predicted > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(k, dtype=float), where=(support + predicted) > 0)
    common = np.isin(classes, common_labels) & supported
    if not common.any():
        raise ValueError('no common supported evaluation classes')
    values = {
        'n_test': len(y_true), 'n_classes_evaluated': int(supported.sum()),
        'n_classes_train': k, 'common_n_classes': int(common.sum()),
        'accuracy': float((y_true == y_pred).mean()),
        'macro_recall': float(recall[supported].mean()),
        'macro_f1': float(f1[supported].mean()),
        'top5_accuracy': float((top == y_true[:, None]).any(axis=1).mean()),
        'common_macro_recall': float(recall[common].mean()),
        'common_macro_f1': float(f1[common].mean()),
        'uniform_chance_macro_recall': 1.0 / k,
        'uniform_chance_top5_accuracy': min(5, k) / k,
    }
    per_class = [{'label': str(classes[i]), 'support': int(support[i]),
                  'predicted': int(predicted[i]), 'correct': int(correct[i]),
                  'recall': float(recall[i]), 'precision': float(precision[i]),
                  'f1': float(f1[i]), 'in_common_set': bool(common[i])} for i in range(k)]
    return values, matrix, per_class


def read_bootstrap(y_true, y_pred, groups, classes, common_labels, repeats, seed):
    """Bayesian cluster bootstrap: one Exp(1) weight per read, shared by windows.

    Equivalent to Dirichlet(1,...,1) read weights for these scale-invariant
    metrics. Positive weights preserve rare-class support in each draw. The
    interval describes test-read reweighting, not probe-refitting uncertainty.
    """
    if repeats < 20:
        raise ValueError('bootstrap_repeats must be at least 20')
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    unique, inverse = np.unique(groups, return_inverse=True)
    shape = (len(unique), len(classes))
    support = coo_matrix((np.ones(len(y_true)), (inverse, y_true)), shape=shape).tocsr()
    correct = coo_matrix(((y_true == y_pred).astype(float), (inverse, y_true)), shape=shape).tocsr()
    available = np.asarray(support.sum(axis=0)).ravel() > 0
    common = available & np.isin(classes, common_labels)
    rng = np.random.default_rng(seed)
    draws, common_draws = [], []
    for _ in range(repeats):
        weights = rng.exponential(1.0, len(unique))
        denominators = np.asarray(support.T @ weights).ravel()
        numerators = np.asarray(correct.T @ weights).ravel()
        recalls = np.divide(numerators, denominators, out=np.zeros(len(classes)), where=denominators > 0)
        draws.append(float(recalls[available].mean()))
        common_draws.append(float(recalls[common].mean()))
    low, high = np.quantile(draws, [0.025, 0.975])
    clow, chigh = np.quantile(common_draws, [0.025, 0.975])
    return {'macro_recall_ci_low': float(low), 'macro_recall_ci_high': float(high),
            'common_macro_recall_ci_low': float(clow), 'common_macro_recall_ci_high': float(chigh),
            'n_test_reads': len(unique), 'interval_method': 'read_cluster_bayesian_bootstrap',
            'bootstrap_repeats': repeats}
