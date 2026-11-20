"""Interchangeable text representations fitted exclusively on training rows."""
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, StandardScaler

TOKEN = re.compile(r"[\w'-]+", re.UNICODE)


def context(row):
    return ' '.join(str(row[f'word{i}']) for i in [-1, 0, 1])


class StaticVectors:
    """Load Word2Vec/fastText text (.vec) vectors; binary files are not assumed."""
    def __init__(self, path):
        self.path = str(path)

    def fit(self, texts):
        self.vectors, self.dimension = {}, None
        with open(self.path, encoding='utf8') as stream:
            for number, line in enumerate(stream):
                fields = line.split()
                if not fields:
                    continue
                if number == 0 and len(fields) == 2 and all(f.isdigit() for f in fields):
                    self.dimension = int(fields[1])
                    continue
                word, *values = fields
                vector = np.asarray(values, dtype=np.float32)
                if self.dimension is None:
                    self.dimension = len(vector)
                if not self.dimension or len(vector) != self.dimension or not np.isfinite(vector).all():
                    raise ValueError(f'Invalid text embedding at {self.path}:{number+1}')
                self.vectors[word.lower()] = vector
        if not self.vectors:
            raise ValueError('No vectors found')
        return self

    def transform(self, texts):
        result = []
        for text in texts:
            tokens = TOKEN.findall(text.lower())
            known = [self.vectors[t] for t in tokens if t in self.vectors]
            vector = np.mean(known, axis=0) if known else np.zeros(self.dimension)
            result.append(np.r_[vector, 1-len(known)/max(1, len(tokens))])
        return np.asarray(result, dtype=np.float32)


class PPMIVectors:
    """Small corpus-trained static baseline: positive PMI + truncated SVD."""
    def fit(self, texts):
        documents = [TOKEN.findall(text.lower()) for text in texts]
        self.vocab = {word: i for i, word in enumerate(sorted({w for d in documents for w in d}))}
        counts = Counter()
        for document in documents:
            for i, word in enumerate(document):
                for j in range(max(0, i-2), min(len(document), i+3)):
                    if j != i:
                        counts[self.vocab[word], self.vocab[document[j]]] += 1
        n = len(self.vocab)
        if n < 2 or not counts:
            self.embeddings = np.ones((n, 1), dtype=np.float32)
            return self
        rows, cols = zip(*counts)
        matrix = sparse.coo_matrix((list(counts.values()), (rows, cols)), shape=(n, n)).tocsr().astype(float)
        rowsum = np.asarray(matrix.sum(axis=1)).ravel()
        colsum = np.asarray(matrix.sum(axis=0)).ravel()
        coo = matrix.tocoo()
        coo.data = np.maximum(0, np.log(coo.data*matrix.sum() / (rowsum[coo.row]*colsum[coo.col])))
        ppmi = coo.tocsr()
        ppmi.eliminate_zeros()
        self.embeddings = normalize(TruncatedSVD(n_components=min(64, n-1), random_state=42).fit_transform(ppmi))
        return self

    def transform(self, texts):
        result = []
        for text in texts:
            tokens = TOKEN.findall(text.lower())
            known = [self.embeddings[self.vocab[t]] for t in tokens if t in self.vocab]
            vector = np.mean(known, axis=0) if known else np.zeros(self.embeddings.shape[1])
            result.append(np.r_[vector, 1-len(known)/max(1, len(tokens))])
        return np.asarray(result, dtype=np.float32)


class FrozenEncoder:
    """Context-window transformer pooling or a SentenceTransformer encoder.

    No embedding encoder is trained on test folds. Downloading is opt-in.
    Heavy runtime objects are reloaded rather than pickled into checkpoints.
    """
    def __init__(self, model, kind='contextual', cache_dir='.cache/hf/hub', local_only=True,
                 batch_size=32, device='cpu'):
        self.model_name, self.kind, self.cache_dir = model, kind, str(cache_dir)
        self.local_only, self.batch_size, self.device = local_only, batch_size, device
        self._model, self._tokenizer, self._cache = None, None, {}

    def fit(self, texts):
        self._load()
        return self

    def _load(self):
        if self._model is not None:
            return
        if self.kind == 'sentence':
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise ImportError('Sentence embeddings require the optional sentence-transformers dependency: install .[embeddings]') from error
            self._model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir,
                                               local_files_only=self.local_only, device=self.device)
        else:
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir,
                                                            local_files_only=self.local_only)
            self._model = AutoModel.from_pretrained(self.model_name, cache_dir=self.cache_dir,
                                                    local_files_only=self.local_only).to(self.device).eval()

    def transform(self, texts):
        self._load()
        missing = sorted(set(texts)-self._cache.keys())
        if self.kind == 'sentence' and missing:
            values = self._model.encode(missing, batch_size=self.batch_size, normalize_embeddings=True,
                                        show_progress_bar=False)
            self._cache.update(zip(missing, values))
        elif missing:
            import torch
            with torch.no_grad():
                for a in range(0, len(missing), self.batch_size):
                    part = missing[a:a+self.batch_size]
                    encoded = self._tokenizer(part, padding=True, truncation=False, return_tensors='pt')
                    if encoded['input_ids'].shape[1] > self._model.config.max_position_embeddings:
                        raise ValueError('Context exceeds the frozen encoder position limit')
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    hidden = self._model(**encoded).last_hidden_state
                    weights = encoded['attention_mask'].unsqueeze(-1)
                    pooled = (hidden*weights).sum(1)/weights.sum(1).clamp(min=1)
                    values = normalize(pooled.cpu().numpy())
                    self._cache.update(zip(part, values))
        return np.asarray([self._cache[text] for text in texts], dtype=np.float32)

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_model=None, _tokenizer=None, _cache={})
        return state


class RepresentationSpace:
    def __init__(self, kind='tfidf', vectors=None, encoder='bert-base-uncased',
                 cache_dir='.cache/hf/hub', local_only=True, batch_size=32, device='cpu'):
        self.kind, self.vectors = kind, vectors
        self.encoder_args = dict(model=encoder, kind=kind, cache_dir=cache_dir,
                                 local_only=local_only, batch_size=batch_size, device=device)

    def _extras(self, rows):
        return [{k: v for k, v in row.items() if not k.startswith('word') and isinstance(v, str)} for row in rows]

    def _numbers(self, rows):
        return [{k: v for k, v in row.items() if isinstance(v, (int, float))} for row in rows]

    def fit(self, rows):
        texts = [context(row) for row in rows]
        self.extras = DictVectorizer().fit(self._extras(rows))
        self.numbers = DictVectorizer().fit(self._numbers(rows))
        values = self.numbers.transform(self._numbers(rows))
        self.numeric_scaler = StandardScaler(with_mean=False).fit(values) if values.shape[1] else None
        if self.kind == 'tfidf':
            self.word = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b[\w'-]+\b").fit(texts)
            self.char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5)).fit(texts)
        elif self.kind == 'ppmi':
            self.text = PPMIVectors().fit(texts)
        elif self.kind == 'static':
            if not self.vectors:
                raise ValueError('Static embeddings require --vectors pointing to a Word2Vec/fastText text file')
            self.text = StaticVectors(self.vectors).fit(texts)
        elif self.kind in ['contextual', 'sentence']:
            self.text = FrozenEncoder(**self.encoder_args).fit(texts)
        else:
            raise ValueError(f'Unknown representation {self.kind}')
        return self

    def transform(self, rows):
        texts = [context(row) for row in rows]
        if self.kind == 'tfidf':
            text = sparse.hstack([self.word.transform(texts), self.char.transform(texts)], format='csr')
        else:
            text = sparse.csr_matrix(self.text.transform(texts))
        parts = [text, self.extras.transform(self._extras(rows))]
        if self.numeric_scaler is not None:
            parts.append(self.numeric_scaler.transform(self.numbers.transform(self._numbers(rows))))
        return sparse.hstack(parts, format='csr')
