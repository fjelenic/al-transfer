import random
from sklearn.model_selection import train_test_split
from datasets import load_dataset


class SubjectivityData:
    def __init__(self):
        self.name = "Subjectivity"
        self.nli = False
        self.num_out = 1
        self.num_iter = 30

    def load(self):
        with open("obj", "r") as f:
            df_obj = [d.strip() for d in f.readlines()]
        with open("sub", "r", encoding="ISO-8859-1") as f:
            df_sub = [d.strip() for d in f.readlines()]

        X_raw = df_obj + df_sub
        y_raw = [0 for _ in range(len(df_obj))] + [1 for _ in range(len(df_sub))]
        random.seed(5)

        inds = list(range(len(X_raw)))
        random.shuffle(inds)
        X = [X_raw[i] for i in inds]
        y = [y_raw[i] for i in inds]

        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=5, stratify=y
        )

        mapping = dict([(text, i) for i, text in enumerate(sorted(train_X))])

        return train_X, test_X, train_y, test_y, mapping


class TrecData:
    def __init__(self):
        self.name = "TREC"
        self.nli = False
        self.num_out = 6
        self.num_iter = 30

    def load(self):
        train = load_dataset("trec", split="train")
        test = load_dataset("trec", split="test")
        mapping = dict([(text, i) for i, text in enumerate(sorted(train["text"]))])
        return (
            train["text"],
            test["text"],
            train["label-coarse"],
            test["label-coarse"],
            mapping,
        )


class AGNewsData:
    def __init__(self):
        self.name = "AG-News"
        self.nli = False
        self.num_out = 4
        self.num_iter = 30

    def load(self):
        train = load_dataset("ag_news", split="train")
        test = load_dataset("ag_news", split="test")

        train_X, train_y = self.subsample(train["text"], train["label"], 20_000)
        mapping = dict([(text, i) for i, text in enumerate(sorted(train_X))])

        return train_X, test["text"], train_y, test["label"], mapping

    def subsample(self, X, y, sample_size):
        train_X, _, train_y, _ = train_test_split(
            X, y, train_size=sample_size, random_state=5, stratify=y
        )
        return train_X, train_y


class CoLAData:
    def __init__(self):
        self.name = "CoLA"
        self.nli = False
        self.num_out = 1
        self.num_iter = 30

    def load(self):
        train = load_dataset("glue", "cola", split="train")
        test = load_dataset("glue", "cola", split="validation")
        mapping = dict([(text, i) for i, text in enumerate(sorted(train["sentence"]))])
        return (
            train["sentence"],
            test["sentence"],
            train["label"],
            test["label"],
            mapping,
        )
