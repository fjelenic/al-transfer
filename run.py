import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import torch
from torch import nn
from scipy import stats
import warnings
import copy
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from torch.autograd import grad
import random
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
import transformers
from my_datasets import SubjectivityData, TrecData, AGNewsData, CoLAData
from my_al import RandomSampler, EntropySampler, CoreSet, BADGE
from my_models import BertClassifier, ElectraClassifier, RobertaClassifier

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_warm_start(X_pool, y_pool, warm_start_size, X_pair_pool=None):
    random_sampler = RandomSampler()
    warm_start_inds = random_sampler.select_batch(
        np.array(X_pool), warm_start_size, X_pair=np.array(X_pair_pool)
    ).tolist()
    warm_start_inds.sort(reverse=True)
    X_labeled = [X_pool.pop(i) for i in warm_start_inds]
    y_labeled = [y_pool.pop(i) for i in warm_start_inds]
    if X_pair_pool:
        X_pair_labeled = [X_pair_pool.pop(i) for i in warm_start_inds]
        return X_labeled, y_labeled, X_pair_labeled
    return X_labeled, y_labeled, None


def fill_labeled_pool(
    inds,
    mapping,
    X_pool,
    y_pool,
    X_labeled,
    y_labeled,
    X_pair_pool=None,
    X_pair_labeled=None,
):
    inds.sort(reverse=True)
    X_labeled.extend([X_pool.pop(i) for i in inds])
    y_labeled.extend([y_pool.pop(i) for i in inds])
    if X_pair_pool:
        X_pair_labeled.extend([X_pair_pool.pop(i) for i in inds])
    return dict(Counter(y_labeled[-len(inds) :])), list(
        map(mapping.get, X_labeled[-len(inds) :])
    )


def load_data(data):
    if data.nli:
        X_pool, X_test, X_pair_pool, X_pair_test, y_pool, y_test, mapping = data.load()
    else:
        X_pool, X_test, y_pool, y_test, mapping = data.load()
        X_pair_pool = X_pair_test = None
    return X_pool, X_test, X_pair_pool, X_pair_test, y_pool, y_test, mapping


def get_al_kwargs(
    X_pool, X_labeled, model, batch_size, criterion, X_pair=None, X_pair_lab=None
):
    al_kwargs = dict(
        X_unlab=X_pool,
        X_lab=X_labeled,
        model=model,
        batch_size=batch_size,
        multilabel=False,
        criterion=criterion,
        X_pair=X_pair,
        X_pair_lab=X_pair_lab,
        device=model.device if model.name in ["BERT", "ELECTRA", "RoBERTa"] else None,
    )
    return al_kwargs


def eval_model(model, X_test, y_test, num_labels, X_pair_test=None):
    if model.name in ["BERT", "ELECTRA", "RoBERTa"]:
        pred = model.predict(X_test, X_pair_test)
    else:
        X_vec = model.vectorize_data(X_test, X_pair_test)
        pred = model.predict(X_vec)
    return pred.reshape(-1).tolist()


if __name__ == "__main__":
    al_methods = [RandomSampler(), EntropySampler(), CoreSet(), BADGE()]
    models = [BertClassifier, ElectraClassifier, RobertaClassifier]
    datasets = [SubjectivityData(), CoLAData(), TrecData(), AGNewsData()]
    model_names = ["BERT", "ELECTRA", "RoBERTa"]
    batch_size = 50

    results = {}
    for data in datasets:
        results[data.name] = {}
        results[data.name]["random"] = {}
        for model in model_names:
            results[data.name]["random"][model] = {}
            results[data.name]["random"][model]["accuracy"] = []
            results[data.name]["random"][model]["labels"] = []
            results[data.name]["random"][model]["examples"] = []
        for al in al_methods[1:]:
            results[data.name][al.name] = {}
            for model_a in model_names:
                results[data.name][al.name][model_a] = {}
                for model_s in model_names:
                    results[data.name][al.name][model_a][model_s] = {}
                    results[data.name][al.name][model_a][model_s]["accuracy"] = []
                    results[data.name][al.name][model_a][model_s]["labels"] = []
                    results[data.name][al.name][model_a][model_s]["examples"] = []

    for data in datasets:
        # RANDOM ACQUISITION
        al = al_methods[0]
        for i, model_cls in enumerate(models):
            for seed in range(20):
                (
                    X_pool,
                    X_test,
                    X_pair_pool,
                    X_pair_test,
                    y_pool,
                    y_test,
                    mapping,
                ) = load_data(data)
                res = []
                selected_labels = []
                selected_examples = []
                set_seed(seed)
                X_labeled, y_labeled, X_pair_labeled = set_warm_start(
                    X_pool, y_pool, batch_size, X_pair_pool
                )
                selected_labels.append(dict(Counter(y_labeled)))
                selected_examples.append(list(map(mapping.get, X_labeled)))
                model = (
                    model_cls(data.num_out, device=torch.device(GPU))
                    if model_names[i] in ["BERT", "ELECTRA", "RoBERTa"]
                    else model_cls()
                )
                if model_names[i] in ["BERT", "ELECTRA", "RoBERTa"]:
                    criterion = (
                        nn.BCEWithLogitsLoss()
                        if data.num_out == 1
                        else nn.CrossEntropyLoss()
                    )
                    criterion.to(model.device)
                else:
                    criterion = None
                model.train_loop(
                    X_labeled,
                    y_labeled,
                    criterion=criterion,
                    X_pair=X_pair_labeled,
                    num_epochs=5,
                )
                res.append(
                    eval_model(
                        model, X_test, y_test, data.num_out, X_pair_test=X_pair_test
                    )
                )
                for iteration in range(data.num_iter):
                    print(
                        f"{data.name}-{al.name}-{model.name}-{seed}-{iteration}/{data.num_iter}"
                    )
                    if model_names[i] in ["BERT", "ELECTRA", "RoBERTa"]:
                        al_kwargs = get_al_kwargs(
                            X_pool,
                            X_labeled,
                            model,
                            batch_size,
                            criterion,
                            X_pair=X_pair_pool,
                            X_pair_lab=X_pair_labeled,
                        )
                    else:
                        X_pool_vec = model.vectorize_data(X_pool, X_pair=X_pair_pool)
                        X_labeled_vec = model.vectorize_data(
                            X_labeled, X_pair=X_pair_labeled
                        )
                        al_kwargs = get_al_kwargs(
                            X_pool_vec, X_labeled_vec, model, batch_size, criterion
                        )
                    inds = al.select_batch(**al_kwargs).tolist()
                    labs_count, example_data = fill_labeled_pool(
                        inds,
                        mapping,
                        X_pool,
                        y_pool,
                        X_labeled,
                        y_labeled,
                        X_pair_pool=X_pair_pool,
                        X_pair_labeled=X_pair_labeled,
                    )
                    selected_labels.append(labs_count)
                    selected_examples.append(example_data)
                    model = (
                        model_cls(data.num_out, device=torch.device(GPU))
                        if model_names[i] in ["BERT", "ELECTRA", "RoBERTa"]
                        else model_cls()
                    )
                    model.train_loop(
                        X_labeled,
                        y_labeled,
                        criterion=criterion,
                        X_pair=X_pair_labeled,
                        num_epochs=5,
                    )
                    res.append(
                        eval_model(
                            model, X_test, y_test, data.num_out, X_pair_test=X_pair_test
                        )
                    )
                results[data.name]["random"][model.name]["accuracy"].append(res)
                results[data.name]["random"][model.name]["labels"].append(
                    selected_labels
                )
                results[data.name]["random"][model.name]["examples"].append(
                    selected_examples
                )
                with open("results.pkl", "wb") as f:
                    pickle.dump(results, f)
                print(f"{data.name}-{al.name}-{model.name}-{seed}: DONE!")

        # AL ACQUISITION
        for al in al_methods[1:]:
            for i, model_a_cls in enumerate(models):
                if model_names[i] == "BERT" or model_names[i] == "ELECTRA":
                    continue
                res = {
                    name: {"accuracy": [[] for _ in range(20)]} for name in model_names
                }
                selected_labels = [[] for _ in range(20)]
                selected_examples = [[] for _ in range(20)]
                for seed in range(20):
                    (
                        X_pool,
                        X_test,
                        X_pair_pool,
                        X_pair_test,
                        y_pool,
                        y_test,
                        mapping,
                    ) = load_data(data)
                    set_seed(seed)
                    X_labeled, y_labeled, X_pair_labeled = set_warm_start(
                        X_pool, y_pool, batch_size, X_pair_pool
                    )
                    selected_labels[seed].append(dict(Counter(y_labeled)))
                    selected_examples[seed].append(list(map(mapping.get, X_labeled)))
                    model_a = (
                        model_a_cls(data.num_out, device=torch.device(GPU))
                        if model_names[i] in ["BERT", "ELECTRA", "RoBERTa"]
                        else model_a_cls()
                    )
                    if model_names[i] in ["BERT", "ELECTRA", "RoBERTa"]:
                        criterion = (
                            nn.BCEWithLogitsLoss()
                            if data.num_out == 1
                            else nn.CrossEntropyLoss()
                        )
                        criterion.to(model_a.device)
                    else:
                        criterion = None
                    model_a.train_loop(
                        X_labeled,
                        y_labeled,
                        criterion=criterion,
                        X_pair=X_pair_labeled,
                        num_epochs=5,
                    )
                    res[model_names[i]]["accuracy"][seed].append(
                        eval_model(
                            model_a,
                            X_test,
                            y_test,
                            data.num_out,
                            X_pair_test=X_pair_test,
                        )
                    )
                    for j, model_s_cls in enumerate(models):
                        if i == j:
                            continue
                        else:
                            model_s = (
                                model_s_cls(data.num_out, device=torch.device(GPU))
                                if model_names[j] in ["BERT", "ELECTRA", "RoBERTa"]
                                else model_s_cls()
                            )
                            if model_names[j] in ["BERT", "ELECTRA", "RoBERTa"]:
                                criterion = (
                                    nn.BCEWithLogitsLoss()
                                    if data.num_out == 1
                                    else nn.CrossEntropyLoss()
                                )
                                criterion.to(model_s.device)
                            else:
                                criterion = None
                            model_s.train_loop(
                                X_labeled,
                                y_labeled,
                                criterion=criterion,
                                X_pair=X_pair_labeled,
                                num_epochs=5,
                            )
                            res[model_names[j]]["accuracy"][seed].append(
                                eval_model(
                                    model_s,
                                    X_test,
                                    y_test,
                                    data.num_out,
                                    X_pair_test=X_pair_test,
                                )
                            )
                    for iteration in range(data.num_iter):
                        print(
                            f"{data.name}-{al.name}-{model_a.name}-{seed}-{iteration}/{data.num_iter}"
                        )
                        if model_names[i] in ["BERT", "ELECTRA", "RoBERTa"]:
                            al_kwargs = get_al_kwargs(
                                X_pool,
                                X_labeled,
                                model_a,
                                batch_size,
                                criterion,
                                X_pair=X_pair_pool,
                                X_pair_lab=X_pair_labeled,
                            )
                        else:
                            X_pool_vec = model_a.vectorize_data(
                                X_pool, X_pair=X_pair_pool
                            )
                            X_labeled_vec = model_a.vectorize_data(
                                X_labeled, X_pair=X_pair_labeled
                            )
                            al_kwargs = get_al_kwargs(
                                X_pool_vec,
                                X_labeled_vec,
                                model_a,
                                batch_size,
                                criterion,
                            )
                        inds = al.select_batch(**al_kwargs).tolist()
                        labs_count, example_data = fill_labeled_pool(
                            inds,
                            mapping,
                            X_pool,
                            y_pool,
                            X_labeled,
                            y_labeled,
                            X_pair_pool=X_pair_pool,
                            X_pair_labeled=X_pair_labeled,
                        )
                        selected_labels[seed].append(labs_count)
                        selected_examples[seed].append(example_data)
                        model_a = (
                            model_a_cls(data.num_out, device=torch.device(GPU))
                            if model_names[i] in ["BERT", "ELECTRA", "RoBERTa"]
                            else model_a_cls()
                        )
                        if model_names[i] in ["BERT", "ELECTRA", "RoBERTa"]:
                            criterion = (
                                nn.BCEWithLogitsLoss()
                                if data.num_out == 1
                                else nn.CrossEntropyLoss()
                            )
                            criterion.to(model_a.device)
                        else:
                            criterion = None
                        model_a.train_loop(
                            X_labeled,
                            y_labeled,
                            criterion=criterion,
                            X_pair=X_pair_labeled,
                            num_epochs=5,
                        )
                        res[model_names[i]]["accuracy"][seed].append(
                            eval_model(
                                model_a,
                                X_test,
                                y_test,
                                data.num_out,
                                X_pair_test=X_pair_test,
                            )
                        )
                        for j, model_s_cls in enumerate(models):
                            if i == j:
                                continue
                            else:
                                model_s = (
                                    model_s_cls(data.num_out, device=torch.device(GPU))
                                    if model_names[j] in ["BERT", "ELECTRA", "RoBERTa"]
                                    else model_s_cls()
                                )
                                if model_names[j] in ["BERT", "ELECTRA", "RoBERTa"]:
                                    criterion = (
                                        nn.BCEWithLogitsLoss()
                                        if data.num_out == 1
                                        else nn.CrossEntropyLoss()
                                    )
                                    criterion.to(model_s.device)
                                else:
                                    criterion = None
                                model_s.train_loop(
                                    X_labeled,
                                    y_labeled,
                                    criterion=criterion,
                                    X_pair=X_pair_labeled,
                                    num_epochs=5,
                                )
                                res[model_names[j]]["accuracy"][seed].append(
                                    eval_model(
                                        model_s,
                                        X_test,
                                        y_test,
                                        data.num_out,
                                        X_pair_test=X_pair_test,
                                    )
                                )
                    print(f"{data.name}-{al.name}-{model_a.name}-{seed}: DONE!")
                results[data.name][al.name][model_a.name] = res
                results[data.name][al.name][model_a.name]["labels"] = selected_labels
                results[data.name][al.name][model_a.name][
                    "examples"
                ] = selected_examples
                with open("results.pkl", "wb") as f:
                    pickle.dump(results, f)

    print("FINISHED!!!")
