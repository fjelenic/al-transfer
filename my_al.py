import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from scipy import stats


class RandomSampler:
    name = "random"

    def select_batch(self, X_unlab, batch_size, X_pair=None, **kwargs):
        if isinstance(X_unlab, list):
            length = len(X_unlab)
        else:
            length = X_unlab.shape[0]
        if length <= batch_size:
            return np.arange(length)

        unlab_inds = np.arange(length)
        return np.random.choice(unlab_inds, size=batch_size, replace=False)

    def get_weights(self, X_unlab, **kwargs):
        return np.ones(len(X_unlab))


class EntropySampler:
    name = "entropy"

    def select_batch(
        self, X_unlab, batch_size, model, multilabel, X_pair=None, **kwargs
    ):
        if isinstance(X_unlab, list):
            length = len(X_unlab)
        else:
            length = X_unlab.shape[0]
        if length <= batch_size:
            return np.arange(length)

        if X_pair:
            probs = model.predict_proba(X_unlab, X_pair)
        else:
            probs = model.predict_proba(X_unlab)

        entropies = self.get_weights(probs, multilabel)

        # Retrieve `batch_size` instances with lowest negative entropies.
        top_n = np.argpartition(entropies, batch_size)[:batch_size]
        return top_n

    def get_weights(self, probs, multilabel, **kwargs):
        # Clip for numerical stability.
        probs = np.clip(probs, a_min=1e-6, a_max=1 - 1e-6)
        if multilabel:
            binary_entropies = -probs * np.log(probs) - (1 - probs) * np.log(1 - probs)
            entropies = np.sum(binary_entropies, axis=1)
        else:
            entropies = np.sum(-probs * np.log(probs), axis=1)

        # We are selecting batch with argmin
        entropies = -entropies
        scaler = MinMaxScaler()
        entropies = scaler.fit_transform(entropies.reshape(-1, 1)).reshape(-1)
        return entropies


class CoreSet:
    """
    Implementation of CoreSet :footcite:`sener2018active` Strategy. A diversity-based
    approach using coreset selection. The embedding of each example is computed by the network’s
    penultimate layer and the samples at each round are selected using a greedy furthest-first
    traversal conditioned on all labeled examples.
    """

    name = "core-set"

    def _furthest_first(self, unlabeled_embeddings, labeled_embeddings, n):
        m = unlabeled_embeddings.shape[0]
        if labeled_embeddings.shape[0] == 0:
            min_dist = torch.tile(float("inf"), m)
        else:
            dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings, p=2)
            min_dist = torch.min(dist_ctr, dim=1)[0]

        idxs = []

        for i in range(n):
            idx = torch.argmax(min_dist)
            idxs.append(idx.item())
            dist_new_ctr = torch.cdist(
                unlabeled_embeddings, unlabeled_embeddings[[idx], :]
            )
            min_dist = torch.minimum(min_dist, dist_new_ctr[:, 0])

        return idxs

    def select_batch(
        self,
        X_unlab,
        X_lab,
        batch_size,
        model,
        device,
        X_pair_unlab=None,
        X_pair_lab=None,
        **kwargs
    ):
        if len(X_unlab) <= batch_size:
            return np.arange(len(X_unlab))

        embedding_unlabeled = model.get_encoded(X_unlab, X_pair_unlab)
        embedding_labeled = model.get_encoded(X_lab, X_pair_lab)

        top_n = self._furthest_first(embedding_unlabeled, embedding_labeled, batch_size)

        return np.array(top_n)


class BADGE:
    """
    This method is based on the paper Deep Batch Active Learning by Diverse, Uncertain Gradient
    Lower Bounds `DBLP-Badge`. According to the paper, this strategy, Batch Active
    learning by Diverse Gradient Embeddings (BADGE), samples groups of points that are disparate
    and high magnitude when represented in a hallucinated gradient space, a strategy designed to
    incorporate both predictive uncertainty and sample diversity into every selected batch.
    Crucially, BADGE trades off between uncertainty and diversity without requiring any hand-tuned
    hyperparameters. Here at each round of selection, loss gradients are computed using the
    hypothesized labels. Then to select the points to be labeled are selected by applying
    k-means++ on these loss gradients.
    """

    name = "badge"

    def _init_centers(self, X, K, device):
        pdist = nn.PairwiseDistance(p=2)
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        inds_all = [ind]
        cent_inds = [0.0] * len(X)
        cent = 0

        while len(mu) < K:
            if len(mu) == 1:
                D2 = pdist(
                    torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device)
                )
                D2 = torch.flatten(D2)
                D2 = D2.cpu().numpy().astype(float)
            else:
                newD = pdist(
                    torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device)
                )
                newD = torch.flatten(newD)
                newD = newD.cpu().numpy().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        cent_inds[i] = cent
                        D2[i] = newD[i]

            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            custom_dist = stats.rv_discrete(
                name="custom", values=(np.arange(len(D2)), Ddist)
            )
            ind = custom_dist.rvs(size=1)[0]
            mu.append(X[ind])
            inds_all.append(ind)
            cent += 1

        return inds_all

    def select_batch(
        self, X_unlab, model, batch_size, criterion, device, X_pair=None, **kwargs
    ):
        if len(X_unlab) <= batch_size:
            return np.arange(len(X_unlab))

        grad_embedding = model.get_grad_embedding(
            X_unlab, criterion, X_pair=X_pair, grad_embedding_type="linear"
        )
        grad_embedding = grad_embedding.cpu().detach().numpy()
        top_n = self._init_centers(grad_embedding, batch_size, device)
        return np.array(top_n)
