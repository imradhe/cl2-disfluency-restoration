"""Sampling of training representations; never interpolate text or token ids."""
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class TrainingSpace:
    """Train-fitted numeric space used only when sampling needs distances."""
    def fit(self, x):
        self.svd = None
        if sparse.issparse(x) and x.shape[1] > 128 and min(x.shape) > 2:
            self.svd = TruncatedSVD(n_components=min(64, min(x.shape)-1), random_state=42)
            dense = self.svd.fit_transform(x)
        else:
            dense = x.toarray() if sparse.issparse(x) else np.asarray(x)
        self.scaler = StandardScaler().fit(dense)
        return self

    def transform(self, x):
        if self.svd is not None:
            dense = self.svd.transform(x)
        else:
            dense = x.toarray() if sparse.issparse(x) else np.asarray(x)
        return self.scaler.transform(dense)


def resample(x, y, policy='none', seed=42):
    """Return representations, targets, source indices, and explicit diagnostics.

    Singleton SMOTE classes use exact random replication (recorded below);
    cluster undersampling selects real rows nearest training-only centroids.
    """
    if policy not in ['none', 'under', 'smote', 'cluster']:
        raise ValueError('Unknown sampling policy')
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    report = {'policy': policy, 'before': dict(Counter(map(str, y))), 'singleton_replication': []}
    if policy == 'none':
        return x, y, np.arange(len(y)), report
    if sparse.issparse(x):
        raise ValueError('Distance-based balancing requires TrainingSpace first')
    target = int(max(counts) if policy == 'smote' else min(counts))
    output_x, output_y, source = [], [], []
    for label, count in zip(classes, counts):
        indices = np.flatnonzero(y == label)
        part = x[indices]
        if policy == 'under':
            chosen = rng.choice(indices, target, replace=False)
            block, origins = x[chosen], chosen
        elif policy == 'cluster':
            if count <= target:
                chosen = indices
            else:
                km = MiniBatchKMeans(n_clusters=target, random_state=seed, n_init=3).fit(part)
                chosen_local = []
                # Keep distinct real rows, even if centroids tie.
                for center in km.cluster_centers_:
                    order = np.argsort(np.sum((part-center)**2, axis=1))
                    chosen_local.append(next(int(i) for i in order if int(i) not in chosen_local))
                chosen = indices[chosen_local]
            block, origins = x[chosen], chosen
        else:
            block, origins = list(part), list(indices)
            if count == 1:
                report['singleton_replication'].append(str(label))
                while len(block) < target:
                    block.append(part[0].copy())
                    origins.append(int(indices[0]))
            else:
                neighbours = NearestNeighbors(n_neighbors=min(6, count)).fit(part).kneighbors(part, return_distance=False)
                while len(block) < target:
                    a = int(rng.integers(count))
                    choices = neighbours[a][neighbours[a] != a]
                    b = int(rng.choice(choices))
                    block.append(part[a] + rng.random()*(part[b]-part[a]))
                    origins.append(int(indices[a]))
            block, origins = np.asarray(block), np.asarray(origins)
        output_x.extend(block)
        output_y.extend([label] * len(block))
        source.extend(origins)
    result_y = np.asarray(output_y)
    report['after'] = dict(Counter(map(str, result_y)))
    return np.asarray(output_x), result_y, np.asarray(source), report
