import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, auc
from scipy.stats import entropy
import my_datasets
from functools import partial
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import spacy
from scipy.optimize import linear_sum_assignment
import pickle

nlp = spacy.load("en_core_web_md")
data_clases = {
    "Subjectivity": my_datasets.SubjectivityData,
    "CoLA": my_datasets.CoLAData,
    "TREC": my_datasets.TrecData,
    "AG-News": my_datasets.AGNewsData,
}
full_accs = {
    "Subjectivity": {"BERT": 0.9625000000000001, "ELECTRA": 0.953, "RoBERTa": 0.963},
    "CoLA": {
        "BERT": 0.8178331735378715,
        "ELECTRA": 0.8638542665388304,
        "RoBERTa": 0.8398849472674976,
    },
    "TREC": {"BERT": 0.968, "ELECTRA": 0.962, "RoBERTa": 0.968},
    "AG-News": {
        "BERT": 0.9280263157894737,
        "ELECTRA": 0.9263157894736842,
        "RoBERTa": 0.9384210526315789,
    },
}


def get_scores(preds, y, method="f1_micro"):
    if method == "f1_micro":
        f = partial(f1_score, average="micro")
    elif method == "f1_macro":
        f = partial(f1_score, average="macro")
    elif method == "f1_binary":
        f = partial(f1_score, average="binary")
    elif method == "accuracy":
        f = accuracy_score

    return np.array([f(pred, y) for pred in preds])


def results_get(results, dataset, al_method, model_a, model_e):
    if al_method == "random":
        return results[dataset]["random"][model_e]
    return results[dataset][al_method][model_a][model_e]


def glove_sim_batch(
    x1, x2, itos, stopping_index, glove_reprs, X, dataset, dist="cos", start=1
):
    dist_func = {"cos": cosine_distances, "euk": euclidean_distances}
    quads = {
        "cos": {
            "Subjectivity": (0.13400485, 0.00818472),
            "CoLA": (0.24785423, 0.012820013),
            "TREC": (0.24610081, 0.013191456),
            "AG-News": (0.15968868, 0.007594059),
        },
        "euk": {
            "Subjectivity": (22.525087, 5.01743),
            "CoLA": (34.200752, 7.3083315),
            "TREC": (32.18843, 6.986838),
            "AG-News": (19.711493, 3.6587567),
        },
    }
    x1 = x1[start : stopping_index + 1]
    x2 = x2[start : stopping_index + 1]

    data = []
    for j in range(len(x1)):
        inds1 = [X.index(itos[i]) for i in x1[j]]
        inds2 = [X.index(itos[i]) for i in x2[j]]
        vec1 = glove_reprs[inds1]
        vec2 = glove_reprs[inds2]

        sim = dist_func[dist](vec1, vec2)
        a, b = linear_sum_assignment(sim)

        data.append(sim[a, b])

    data = np.array(data)

    micro_avg = data.mean(axis=1)
    micro_std = data.std(axis=1)
    q_min, q_max = (
        quads[dist][dataset][0] - quads[dist][dataset][1],
        quads[dist][dataset][0] + quads[dist][dataset][1],
    )
    minimum = (data < q_min).mean(axis=1)
    maximum = (data > q_max).mean(axis=1)
    none = ((data >= q_min) & (data <= q_max)).mean(axis=1)

    return micro_avg, micro_std, minimum, maximum, none


def glove_sim_accum(
    x1, x2, itos, stopping_index, glove_reprs, X, dataset, dist="cos", start=1
):
    quads = {
        "cos": {
            "Subjectivity": (0.13400485, 0.00818472),
            "CoLA": (0.24785423, 0.012820013),
            "TREC": (0.24610081, 0.013191456),
            "AG-News": (0.15968868, 0.007594059),
        },
        "euk": {
            "Subjectivity": (22.525087, 5.01743),
            "CoLA": (34.200752, 7.3083315),
            "TREC": (32.18843, 6.986838),
            "AG-News": (19.711493, 3.6587567),
        },
    }
    dist_func = {"cos": cosine_distances, "euk": euclidean_distances}
    x1 = x1[start : stopping_index + 1]
    x2 = x2[start : stopping_index + 1]

    q_min, q_max = (
        quads[dist][dataset][0] - quads[dist][dataset][1],
        quads[dist][dataset][0] + quads[dist][dataset][1],
    )
    micro_avg = []
    micro_std = []
    minimum = []
    maximum = []
    none = []
    inds1 = []
    inds2 = []
    for j in range(len(x1)):
        inds1.extend([X.index(itos[i]) for i in x1[j]])
        inds2.extend([X.index(itos[i]) for i in x2[j]])
        vec1 = glove_reprs[inds1]
        vec2 = glove_reprs[inds2]

        sim = dist_func[dist](vec1, vec2)
        a, b = linear_sum_assignment(sim)

        data = sim[a, b]

        micro_avg.append(data.mean())
        micro_std.append(data.std())
        minimum.append((data < q_min).mean())
        maximum.append((data > q_max).mean())
        none.append(((data >= q_min) & (data <= q_max)).mean())

    return micro_avg, micro_std, minimum, maximum, none


def diversity(B, dist="cos"):
    dist_func = {"cos": cosine_distances, "euk": euclidean_distances}
    dists = dist_func[dist](B)
    np.fill_diagonal(dists, np.nan)
    min_d = np.nanmin(dists, axis=1)
    return min_d.mean()


def representativeness(B, U, dist="cos", k=10):
    dist_func = {"cos": cosine_distances, "euk": euclidean_distances}
    dists = dist_func[dist](B, U)
    mins = np.partition(dists, k, axis=1)[:, :k]
    return 1 / mins.mean(axis=1).mean()


def div_repr(X_train, X, glove_reprs, stopping_index, itos, dist="cos", k=10, start=1):
    init = [item for sublist in X[:start] for item in sublist]

    divs = []
    repres = []
    for i in range(start, stopping_index + 1):
        inds = [X.index(itos[j]) for j in X_train[i]]
        B = glove_reprs[inds]
        U = glove_reprs[[j for j in range(len(glove_reprs)) if j not in init]]
        divs.append(diversity(B, dist=dist))
        repres.append(representativeness(B, U, dist=dist, k=k))

        init.extend(inds)

    return divs, repres


def flatten(l):
    return [item for sublist in l for item in sublist]


def main(results, data_clases, full_accs, columns):
    num_seeds = 20
    df = []
    for dataset in ["Subjectivity", "CoLA", "AG-News", "TREC"]:
        dataset_object = data_clases[dataset]()
        X_train, X_test, y_train, y_test, stoi = dataset_object.load()
        itos = {i: s for s, i in stoi.items()}
        glove_reprs = np.array([nlp(x).vector for x in X_train])
        for al_method in ["random", "entropy", "core-set", "badge"]:
            for model_e in ["BERT", "ELECTRA", "RoBERTa"]:
                for model_a in ["BERT", "ELECTRA", "RoBERTa"]:
                    for seed in range(num_seeds):
                        if al_method == "random" and model_a != model_e:
                            continue

                        print(f"{dataset}-{al_method}--{model_e}--{model_a}--{seed}")

                        data_row = [
                            dataset,
                            al_method,
                            model_a if al_method != "random" else None,
                            model_e,
                            seed,
                        ]
                        num_labels = (
                            dataset_object.num_out + 1
                            if dataset_object.num_out == 1
                            else dataset_object.num_out
                        )
                        data_row.extend(
                            [
                                dataset_object.num_iter,
                                num_labels,
                                full_accs[dataset][model_e],
                            ]
                        )

                        labs_distribution = Counter(y_train)
                        data_row.append(entropy(list(labs_distribution.values())))

                        info = results_get(
                            results, dataset, al_method, model_a, model_e
                        )
                        scores = get_scores(info["accuracy"][seed], y_test)
                        steps = [i * 50 for i in range(len(scores))]

                        stopping_index = 30 - 1
                        data_row.append(stopping_index)

                        scores_stop = scores[: stopping_index + 1]
                        steps_stop = steps[: stopping_index + 1]

                        # Dependant variables
                        data_row.append(auc(steps_stop, scores_stop) / steps_stop[-1])
                        data_row.append(scores_stop[-1])

                        for i in range(stopping_index + 1):
                            try:
                                data_row.append(
                                    auc(steps_stop[i:], scores_stop[i:])
                                    / steps_stop[-(i + 1)]
                                )
                            except:
                                data_row.append(None)

                        for i in range(stopping_index + 1):
                            data_row.append(scores_stop[i])

                        if al_method == "random":
                            data_row.extend([None] * (len(data_row) - len(columns)))
                            df.append(data_row)
                            continue

                        for dist in ["cos", "euk"]:
                            (
                                avg_random_a,
                                std_random_a,
                                min_random_a,
                                max_random_a,
                                none_random_a,
                            ) = glove_sim_accum(
                                r[dataset][al_method][model_a]["examples"][seed],
                                r[dataset]["random"][model_a]["examples"][seed],
                                itos,
                                stopping_index,
                                glove_reprs,
                                X_train,
                                dataset,
                                dist=dist,
                                start=0,
                            )
                            for i in range(30):
                                data_row.append(avg_random_a[i])
                                data_row.append(std_random_a[i])
                                data_row.append(min_random_a[i])
                                data_row.append(max_random_a[i])
                                data_row.append(none_random_a[i])
                        for dist in ["cos", "euk"]:
                            (
                                avg_a_e,
                                std_a_e,
                                min_a_e,
                                max_a_e,
                                none_a_e,
                            ) = glove_sim_accum(
                                r[dataset][al_method][model_a]["examples"][seed],
                                r[dataset][al_method][model_e]["examples"][seed],
                                itos,
                                stopping_index,
                                glove_reprs,
                                X_train,
                                dataset,
                                dist=dist,
                                start=0,
                            )
                            for i in range(30):
                                data_row.append(avg_a_e[i])
                                data_row.append(std_a_e[i])
                                data_row.append(min_a_e[i])
                                data_row.append(max_a_e[i])
                                data_row.append(none_a_e[i])

                        for dist in ["cos", "euk"]:
                            (
                                avg_random_a,
                                std_random_a,
                                min_random_a,
                                max_random_a,
                                none_random_a,
                            ) = glove_sim_batch(
                                r[dataset][al_method][model_a]["examples"][seed],
                                r[dataset]["random"][model_a]["examples"][seed],
                                itos,
                                stopping_index,
                                glove_reprs,
                                X_train,
                                dataset,
                                dist=dist,
                                start=0,
                            )
                            for i in range(30):
                                data_row.append(avg_random_a[i])
                                data_row.append(std_random_a[i])
                                data_row.append(min_random_a[i])
                                data_row.append(max_random_a[i])
                                data_row.append(none_random_a[i])
                        for dist in ["cos", "euk"]:
                            (
                                avg_a_e,
                                std_a_e,
                                min_a_e,
                                max_a_e,
                                none_a_e,
                            ) = glove_sim_batch(
                                r[dataset][al_method][model_a]["examples"][seed],
                                r[dataset][al_method][model_e]["examples"][seed],
                                itos,
                                stopping_index,
                                glove_reprs,
                                X_train,
                                dataset,
                                dist=dist,
                                start=0,
                            )
                            for i in range(30):
                                data_row.append(avg_a_e[i])
                                data_row.append(std_a_e[i])
                                data_row.append(min_a_e[i])
                                data_row.append(max_a_e[i])
                                data_row.append(none_a_e[i])

                        for dist in ["cos", "euk"]:
                            divs, repres = div_repr(
                                r[dataset][al_method][model_a]["examples"][seed],
                                X_train,
                                glove_reprs,
                                stopping_index,
                                itos,
                                dist=dist,
                                k=10,
                                start=0,
                            )
                            for i in range(30):
                                data_row.append(divs[i])
                                data_row.append(repres[i])

                        df.append(data_row)

    return pd.DataFrame(data=df, columns=columns)


if __name__ == "__main__":
    with open("results.pkl", "rb") as f:
        r = pickle.load(f)

    columns = [
        "dataset",
        "al_method",
        "model_a",
        "model_e",
        "seed",
        "num_steps",
        "num_labels",
        "full_accuracy",
        "dataset_labels_entropy",
        "stopping_index",
        "auc",
        "f1",
    ]
    columns += [f"auc_{i}" for i in range(30)]
    columns += [f"f1_{i}" for i in range(30)]
    columns.extend(
        flatten(
            [
                [
                    s + f"_{i}"
                    for s in [
                        "cos_glove_micro_avg_random_a_accum",
                        "cos_glove_micro_std_random_a_accum",
                        "cos_glove_micro_min_random_a_accum",
                        "cos_glove_micro_max_random_a_accum",
                        "cos_glove_micro_none_random_a_accum",
                    ]
                ]
                for i in range(30)
            ]
        )
    )
    columns.extend(
        flatten(
            [
                [
                    s + f"_{i}"
                    for s in [
                        "euk_glove_micro_avg_random_a_accum",
                        "euk_glove_micro_std_random_a_accum",
                        "euk_glove_micro_min_random_a_accum",
                        "euk_glove_micro_max_random_a_accum",
                        "euk_glove_micro_none_random_a_accum",
                    ]
                ]
                for i in range(30)
            ]
        )
    )
    columns.extend(
        flatten(
            [
                [
                    s + f"_{i}"
                    for s in [
                        "cos_glove_micro_avg_a_e_accum",
                        "cos_glove_micro_std_a_e_accum",
                        "cos_glove_micro_min_a_e_accum",
                        "cos_glove_micro_max_a_e_accum",
                        "cos_glove_micro_none_a_e_accum",
                    ]
                ]
                for i in range(30)
            ]
        )
    )
    columns.extend(
        flatten(
            [
                [
                    s + f"_{i}"
                    for s in [
                        "euk_glove_micro_avg_a_e_accum",
                        "euk_glove_micro_std_a_e_accum",
                        "euk_glove_micro_min_a_e_accum",
                        "euk_glove_micro_max_a_e_accum",
                        "euk_glove_micro_none_a_e_accum",
                    ]
                ]
                for i in range(30)
            ]
        )
    )

    columns.extend(
        flatten(
            [
                [
                    s + f"_{i}"
                    for s in [
                        "cos_glove_micro_avg_random_a",
                        "cos_glove_micro_std_random_a",
                        "cos_glove_micro_min_random_a",
                        "cos_glove_micro_max_random_a",
                        "cos_glove_micro_none_random_a",
                    ]
                ]
                for i in range(30)
            ]
        )
    )
    columns.extend(
        flatten(
            [
                [
                    s + f"_{i}"
                    for s in [
                        "euk_glove_micro_avg_random_a",
                        "euk_glove_micro_std_random_a",
                        "euk_glove_micro_min_random_a",
                        "euk_glove_micro_max_random_a",
                        "euk_glove_micro_none_random_a",
                    ]
                ]
                for i in range(30)
            ]
        )
    )
    columns.extend(
        flatten(
            [
                [
                    s + f"_{i}"
                    for s in [
                        "cos_glove_micro_avg_a_e",
                        "cos_glove_micro_std_a_e",
                        "cos_glove_micro_min_a_e",
                        "cos_glove_micro_max_a_e",
                        "cos_glove_micro_none_a_e",
                    ]
                ]
                for i in range(30)
            ]
        )
    )
    columns.extend(
        flatten(
            [
                [
                    s + f"_{i}"
                    for s in [
                        "euk_glove_micro_avg_a_e",
                        "euk_glove_micro_std_a_e",
                        "euk_glove_micro_min_a_e",
                        "euk_glove_micro_max_a_e",
                        "euk_glove_micro_none_a_e",
                    ]
                ]
                for i in range(30)
            ]
        )
    )

    columns.extend(
        flatten(
            [
                [s + f"_{i}" for s in ["diversity_cos", "representativeness_cos"]]
                for i in range(30)
            ]
        )
    )
    columns.extend(
        flatten(
            [
                [s + f"_{i}" for s in ["diversity_euk", "representativeness_euk"]]
                for i in range(30)
            ]
        )
    )

    df = main(r, data_clases, full_accs, columns)
    df.to_csv(f"data.csv")
