import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import random
import math
from torch.autograd import grad


def get_vectorizer(arh, device):
    tokenizer = AutoTokenizer.from_pretrained(arh)

    def func(x, x_pair=None):
        if x_pair:
            return tokenizer(
                x, x_pair, truncation=True, padding=True, return_tensors="pt"
            ).to(device)
        else:
            return tokenizer(x, truncation=True, padding=True, return_tensors="pt").to(
                device
            )

    return func


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        output_dim,
        name,
        arh,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.name = name
        self.device = device
        self.multilabel = False
        self.output_dim = output_dim
        self.bert = AutoModel.from_pretrained(arh)
        self.vectorizer = get_vectorizer(arh, self.device)
        config = self.bert.config
        self.out = nn.Linear(config.hidden_size, output_dim)
        self.to(self.device)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, X, X_pair=None):
        if X_pair:
            X_vec = self.vectorizer(list(X), list(X_pair))
        else:
            X_vec = self.vectorizer(list(X))
        out = self.bert(**X_vec)
        logits = self.out(out.last_hidden_state[:, 0, :])
        return logits

    def get_encoded(self, X, X_pair=None):
        embeddings = []
        for X_iter, y_iter, X_pair_iter in self.iterator(
            64 if data.name != "AG-News" else 32, X, None, X_pair=X_pair, shuffle=False
        ):
            if X_pair_iter:
                X_vec = self.vectorizer(list(X_iter), list(X_pair_iter))  # .to("cpu")
            else:
                X_vec = self.vectorizer(list(X_iter))  # .to("cpu")
            out = self.bert(**X_vec)
            embeddings.extend(out.last_hidden_state[:, 0, :].tolist())
        return torch.tensor(embeddings)

    def get_grad_embedding(
        self, X, criterion, X_pair=None, grad_embedding_type="bias_linear"
    ):
        criterion = copy.deepcopy(criterion)
        criterion.reduction = "none"
        criterion.to(self.device)
        grad_embeddings = []

        for X_iter, y_iter, X_pair_iter in self.iterator(
            32 if data.name != "AG-News" else 16, X, None, X_pair=X_pair, shuffle=False
        ):
            self.zero_grad()

            logits = self(X_iter, X_pair_iter)

            if logits.shape[1] == 1 or self.multilabel:
                logits = logits.ravel()
                y = torch.as_tensor(
                    logits > 0,
                    dtype=torch.float,
                    device=self.device,
                )
            else:
                y = torch.argmax(logits, dim=1)

            loss = criterion(logits, y)
            for l in loss:
                if grad_embedding_type == "bias":
                    embedding = grad(l, self.out.bias, retain_graph=True)[0]
                elif grad_embedding_type == "linear":
                    embedding = grad(l, self.out.weight, retain_graph=True)[0]
                elif grad_embedding_type == "bias_linear":
                    weight_emb = grad(l, self.out.weight, retain_graph=True)[0]
                    bias_emb = grad(l, self.out.bias, retain_graph=True)[0]
                    embedding = torch.cat(
                        (weight_emb, bias_emb.unsqueeze(dim=1)), dim=1
                    )
                else:
                    raise ValueError(
                        f"Grad embedding type '{grad_embedding_type}' not supported."
                        "Viable options: 'bias', 'linear', or 'bias_linear'"
                    )
                grad_embeddings.append(embedding.flatten().to("cpu"))

        return torch.stack(grad_embeddings)

    def train_loop(
        self, X, y, criterion, X_pair=None, num_epochs=5, batch_size=64, **kwargs
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        self.train()
        for _ in range(num_epochs):
            # 5 step training routine
            # ------------------------------------------
            for X_iter, y_iter, X_pair_iter in self.iterator(
                batch_size, X, y, X_pair=X_pair
            ):
                # 1) zero the gradients
                optimizer.zero_grad()

                # 2) compute the output
                if X_pair_iter:
                    y_pred = self(X_iter, X_pair_iter)
                else:
                    y_pred = self(X_iter)
                y_iter = torch.tensor(y_iter).to(self.device)
                if y_pred.shape[1] == 1 or self.multilabel:
                    y_iter = y_iter.float()

                # 3) compute the loss
                loss = criterion(y_pred.squeeze(), y_iter.squeeze())

                # 4) use loss to produce gradients
                loss.backward()

                # 5) use optimizer to take gradient step
                optimizer.step()

    def _predict_proba(self, X, X_pair=None):
        self.eval()
        y_pred = []
        for X_iter, y_iter, X_pair_iter in self.iterator(
            32, X, None, X_pair=X_pair, shuffle=False
        ):
            y_pred.extend(self.forward(X_iter, X_pair_iter).tolist())
        y_pred = torch.tensor(y_pred)
        if self.output_dim == 1:
            y_pred = torch.sigmoid(y_pred)
            y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)
        elif self.multilabel:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = F.softmax(y_pred, dim=1)
        self.train()
        return y_pred

    def _predict(self, X, X_pair=None):
        self.eval()
        y_pred = []
        for X_iter, y_iter, X_pair_iter in self.iterator(
            32, X, None, X_pair=X_pair, shuffle=False
        ):
            y_pred.extend(self.forward(X_iter, X_pair_iter).tolist())
        y_pred = torch.tensor(y_pred)
        if self.output_dim == 1 or self.multilabel:
            out = torch.as_tensor(
                y_pred > 0,
                dtype=torch.long,
                device=self.device,
            )
        else:
            out = torch.argmax(y_pred, dim=1)

        self.train()
        return out

    def predict_proba(self, X, X_pair=None):
        if not isinstance(self, BertClassifier):
            X = torch.tensor(X, device=self.device)
        with torch.no_grad():
            y_pred = self._predict_proba(X, X_pair)
            return y_pred.cpu().numpy()

    def predict(self, X, X_pair=None):
        if not isinstance(self, BertClassifier):
            X = torch.tensor(X, device=self.device)
        with torch.no_grad():
            out = self._predict(X, X_pair)
            return out.cpu().numpy()

    def iterator(self, batch_size, X, y=None, X_pair=None, shuffle=True):
        inds = list(range(len(X)))
        if shuffle:
            random.shuffle(inds)
        X_temp = [X[i] for i in inds]
        if y:
            y_temp = [y[i] for i in inds]
        if X_pair:
            X_pair_temp = [X_pair[i] for i in inds]
        for i in range(math.ceil(len(X) / batch_size)):
            X_pair_yield = (
                X_pair_temp[i * batch_size : (i + 1) * batch_size] if X_pair else None
            )
            y_yield = y_temp[i * batch_size : (i + 1) * batch_size] if y else None
            yield X_temp[i * batch_size : (i + 1) * batch_size], y_yield, X_pair_yield


class BertClassifier(TransformerClassifier):
    def __init__(
        self,
        output_dim,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(
            output_dim,
            "BERT",
            "bert-base-uncased",
            device=device,
        )


class ElectraClassifier(nn.Module):
    def __init__(
        self,
        output_dim,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(
            output_dim,
            "ELECTRA",
            "google/electra-base-discriminator",
            device=device,
        )


class RobertaClassifier(nn.Module):
    def __init__(
        self,
        output_dim,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(
            output_dim,
            "RoBERTa",
            "roberta-base",
            device=device,
        )
