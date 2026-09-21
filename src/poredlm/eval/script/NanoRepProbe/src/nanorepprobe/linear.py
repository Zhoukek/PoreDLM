"""Deterministic, class/domain balanced linear ridge probes (no encoder training)."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh


@dataclass
class RidgeSystem:
    mean: np.ndarray
    scale: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    rotated_rhs: np.ndarray
    target_mean: np.ndarray
    classes: np.ndarray
    n_samples: int
    constant_features: int

    def coefficients(self, regularization):
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError('regularization must be finite and positive')
        coef = self.eigenvectors @ (self.rotated_rhs / (self.eigenvalues[:, None] + regularization))
        return coef

    def predict_scores(self, features, coef):
        return ((np.asarray(features, dtype=np.float64) - self.mean) / self.scale) @ coef + self.target_mean


def fit_system(features, labels, domains, classes):
    """Minimize sum_i w_i ||y_i - b - standardized(x_i) W||² + lambda ||W||².

    Sum w=1, each source domain has equal weight and its classes have equal
    weight. Standardization uses these source TRAIN weights only; the intercept
    is unpenalized. Classes must be supported in each participating domain.
    """
    x = np.asarray(features, dtype=np.float64)
    labels, domains, classes = np.asarray(labels), np.asarray(domains), np.asarray(classes)
    if x.ndim != 2 or x.shape[0] != len(labels) or len(domains) != len(labels):
        raise ValueError('incompatible training dimensions')
    if not np.isfinite(x).all() or not len(x):
        raise ValueError('training features must be nonempty and finite')
    if len(classes) < 2 or len(np.unique(classes)) != len(classes):
        raise ValueError('at least two unique output classes required')
    mapping = {label: i for i, label in enumerate(classes)}
    if any(label not in mapping for label in labels):
        raise ValueError('training label not in output vocabulary')
    y = np.array([mapping[label] for label in labels], dtype=np.int64)
    unique_domains = np.unique(domains)
    weights = np.empty(len(y), dtype=np.float64)
    for domain in unique_domains:
        mask = domains == domain
        counts = np.bincount(y[mask], minlength=len(classes))
        if (counts == 0).any():
            raise ValueError(f'source domain {domain} has missing training classes')
        weights[mask] = 1.0 / (len(unique_domains) * len(classes) * counts[y[mask]])
    mean = weights @ x
    centered = x - mean
    variance = weights @ (centered * centered)
    scale = np.sqrt(variance)
    constant = scale <= 1e-12
    scale[constant] = 1.0
    z = centered / scale
    weighted_z = z * weights[:, None]
    gram = z.T @ weighted_z
    # One-hot class targets are accumulated without allocating an N x K array.
    rhs = np.zeros((len(classes), x.shape[1]), dtype=np.float64)
    np.add.at(rhs, y, weighted_z)
    target_mean = np.bincount(y, weights=weights, minlength=len(classes))
    rhs = rhs.T - (weights @ z)[:, None] * target_mean[None, :]
    values, vectors = eigh(gram, check_finite=False, driver='evd')
    values = np.maximum(values, 0.0)
    return RidgeSystem(mean, scale, values, vectors, vectors.T @ rhs,
                       target_mean, classes, len(y), int(constant.sum()))


def predict(system, features, coef, batch_size=2048):
    """Return output-class indices and top min(5,K) indices, never probabilities."""
    predictions, top = [], []
    for start in range(0, len(features), batch_size):
        scores = system.predict_scores(features[start:start + batch_size], coef)
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite classifier scores')
        # Stable sorting fixes tie handling by vocabulary order.
        ranked = np.argsort(-scores, axis=1, kind='stable')[:, :min(5, len(system.classes))]
        predictions.append(ranked[:, 0])
        top.append(ranked)
    if not predictions:
        return np.empty(0, dtype=np.int64), np.empty((0, min(5, len(system.classes))), dtype=np.int64)
    return np.concatenate(predictions), np.concatenate(top)
