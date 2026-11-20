"""BERT BIO tagger plus insertion-slot classifier, with optional numeric features."""
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, BertConfig, BertModel

from .baselines import restrict_predictions
from .alignment import phone_evidence
from .data import BIO, NONE, targets
from .features import LINGUISTIC
from .sampling import TrainingSpace, resample

MARKERS = ['[GAP]', '[WORD]', '[END]', '[PHONES]']


class RestorationBert(nn.Module):
    def __init__(self, encoder, classes, numeric_dim):
        super().__init__()
        self.encoder = encoder
        size = encoder.config.hidden_size
        self.numeric = nn.Linear(numeric_dim, size, bias=False) if numeric_dim else None
        self.dropout = nn.Dropout(.1)
        self.classifier = nn.Linear(size, classes)

    def forward(self, batch, hidden_states=False):
        result = self.encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                              output_hidden_states=hidden_states)
        hidden = result.last_hidden_state[batch['entity_batch'], batch['entity_pos']]
        combined = hidden
        if self.numeric is not None:
            combined = combined + self.numeric(batch['numeric'])
        return {'logits': self.classifier(self.dropout(combined)), 'embeddings': combined,
                'hidden_states': result.hidden_states}


class BertTagger:
    def __init__(self, builder, encoder='bert-base-uncased', cache_dir='.cache/hf/hub',
                 tiny=False, local_only=True, epochs=3, batch_size=8, learning_rate=2e-5,
                 balancing='none', seed=42, max_length=512, device='cpu'):
        self.builder, self.encoder_name, self.cache_dir = builder, encoder, cache_dir
        self.tiny, self.local_only = tiny, local_only
        self.epochs, self.batch_size, self.learning_rate = epochs, batch_size, learning_rate
        self.balancing, self.seed, self.max_length = balancing, seed, max_length
        self.device = torch.device(device)
        self.balanced_head = None

    def _encode(self, ex, labelled=False):
        ids = [self.tokenizer.cls_token_id]
        wp, gp = [], []
        n = len(ex['words'])
        evidence = phone_evidence([w['canonical'] for w in ex['words']], ex['observed_phones']) if self.builder.condition in 'DE' else None
        for i in range(n):
            gp.append(len(ids)); ids.append(self.marker_ids['[GAP]'])
            wp.append(len(ids)); ids.append(self.marker_ids['[WORD]'])
            text = self.builder.serialize(ex, i, 'word', evidence=evidence)
            ids.extend(self.tokenizer.encode(text, add_special_tokens=False))
        gp.append(len(ids)); ids.append(self.marker_ids['[GAP]'])
        ids.extend(self.tokenizer.encode(self.builder.serialize(ex, n, 'gap', evidence=evidence), add_special_tokens=False))
        if self.builder.condition == 'C':
            ids.append(self.marker_ids['[PHONES]'])
            ids.extend(self.tokenizer.encode(' '.join(ex['observed_phones']), add_special_tokens=False))
        ids.append(self.tokenizer.sep_token_id)
        if len(ids) > self.max_length:
            raise ValueError(f"{ex['id']}: {len(ids)} BERT tokens exceed {self.max_length}; rebuild with smaller --max-words (no silent truncation)")
        rows = self.builder.rows(ex)
        numeric = np.array([[r.get(k, 0.) for k in LINGUISTIC] for r in rows], dtype=np.float32)
        if self.builder.condition == 'E':
            numeric = (numeric-self.numeric_mean)/self.numeric_std
        else:
            numeric = np.zeros((len(rows), 0), np.float32)
        labels = [self.class_ids[t] for t in targets(ex)] if labelled else None
        return {'ids': ids, 'positions': wp+gp, 'numeric': numeric, 'labels': labels,
                'tasks': ['word']*n + ['gap']*(n+1)}

    def _batch(self, items):
        n = max(len(item['ids']) for item in items)
        ids = torch.full((len(items), n), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        batches, positions, numeric, tasks, labels = [], [], [], [], []
        for i, item in enumerate(items):
            ids[i, :len(item['ids'])] = torch.tensor(item['ids'])
            mask[i, :len(item['ids'])] = 1
            batches.extend([i]*len(item['positions']))
            positions.extend(item['positions'])
            numeric.extend(item['numeric'])
            tasks.extend(item['tasks'])
            if item['labels'] is not None:
                labels.extend(item['labels'])
        allowed = [[c.startswith(task+':') for c in self.classes_] for task in tasks]
        result = {'input_ids': ids, 'attention_mask': mask,
                  'entity_batch': torch.tensor(batches), 'entity_pos': torch.tensor(positions),
                  'numeric': torch.from_numpy(np.asarray(numeric, dtype=np.float32)),
                  'allowed': torch.tensor(allowed, dtype=torch.bool)}
        if labels:
            result['labels'] = torch.tensor(labels)
        return {key: value.to(self.device) for key, value in result.items()}

    def fit(self, examples):
        torch.manual_seed(self.seed)
        self.builder.fit(examples)
        self.classes_ = np.array(sorted({'word:'+b for b in BIO} | {'gap:'+NONE} |
                                       {t for ex in examples for t in targets(ex) if t.startswith('gap:')}))
        self.class_ids = {c: i for i, c in enumerate(self.classes_)}
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name, cache_dir=self.cache_dir,
                                                       local_files_only=self.local_only)
        self.tokenizer.add_special_tokens({'additional_special_tokens': MARKERS})
        self.marker_ids = {m: self.tokenizer.convert_tokens_to_ids(m) for m in MARKERS}
        if self.tiny:
            config = BertConfig(vocab_size=len(self.tokenizer), hidden_size=64, num_hidden_layers=2,
                                num_attention_heads=4, intermediate_size=128, max_position_embeddings=self.max_length)
            encoder = BertModel(config)
        else:
            encoder = BertModel.from_pretrained(self.encoder_name, cache_dir=self.cache_dir,
                                                 local_files_only=self.local_only)
            encoder.resize_token_embeddings(len(self.tokenizer))
        self.numeric_mean = np.zeros(len(LINGUISTIC), np.float32)
        self.numeric_std = np.ones(len(LINGUISTIC), np.float32)
        if self.builder.condition == 'E':
            values = np.array([[r.get(k, 0.) for k in LINGUISTIC] for ex in examples for r in self.builder.rows(ex)])
            self.numeric_mean = values.mean(axis=0).astype(np.float32)
            self.numeric_std = np.maximum(values.std(axis=0), 1e-6).astype(np.float32)
        self.network = RestorationBert(encoder, len(self.classes_), len(LINGUISTIC) if self.builder.condition == 'E' else 0).to(self.device)
        items = [self._encode(ex, labelled=True) for ex in examples]
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(self.seed)
        self.history = []
        self.network.train()
        for epoch in range(self.epochs):
            order, loss_sum, steps = rng.permutation(len(items)), 0., 0
            for a in range(0, len(order), self.batch_size):
                batch = self._batch([items[int(i)] for i in order[a:a+self.batch_size]])
                optimizer.zero_grad()
                logits = self.network(batch)['logits'].masked_fill(~batch['allowed'], -1e4)
                loss = nn.functional.cross_entropy(logits, batch['labels'])
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
                optimizer.step()
                loss_sum += float(loss.detach()); steps += 1
            self.history.append({'epoch': epoch+1, 'loss': loss_sum/max(1, steps)})
        self.sampling_report = {'policy': 'none', 'encoder_initialization': 'random_tiny' if self.tiny else 'pretrained',
                                'history': self.history}
        if self.balancing != 'none':
            # SMOTE is defined on continuous representations, never on token ids.
            from sklearn.linear_model import LogisticRegression
            embeddings, tasks, _ = self.representations(examples)
            self.balance_space = TrainingSpace().fit(embeddings)
            features = self.balance_space.transform(embeddings)
            labels = np.asarray([t for ex in examples for t in targets(ex)])
            features, labels, _, report = resample(features, labels, self.balancing, self.seed)
            self.balanced_head = LogisticRegression(max_iter=500, random_state=self.seed).fit(features, labels)
            self.sampling_report.update(report, stage='train-fold frozen-encoder entity representations')
        return self

    @torch.no_grad()
    def representations(self, examples, all_layers=False):
        self.network.eval()
        features, tasks, layer_values = [], [], None
        for a in range(0, len(examples), self.batch_size):
            items = [self._encode(ex) for ex in examples[a:a+self.batch_size]]
            batch = self._batch(items)
            result = self.network(batch, hidden_states=all_layers)
            features.append(result['embeddings'].cpu().numpy())
            tasks.extend(task for item in items for task in item['tasks'])
            if all_layers:
                if layer_values is None:
                    layer_values = [[] for _ in result['hidden_states']]
                for i, hidden in enumerate(result['hidden_states']):
                    selected = hidden[batch['entity_batch'], batch['entity_pos']].cpu().numpy()
                    layer_values[i].append(selected)
        return np.concatenate(features), tasks, [np.concatenate(v) for v in layer_values] if all_layers else None

    @torch.no_grad()
    def predict(self, examples):
        embeddings, tasks, _ = self.representations(examples)
        if self.balanced_head is not None:
            probabilities = self.balanced_head.predict_proba(self.balance_space.transform(embeddings))
            classes = self.balanced_head.classes_
        else:
            # eval() above disables dropout; the same deterministic final embeddings feed the head.
            logits = self.network.classifier(torch.from_numpy(embeddings).to(self.device)).cpu().numpy()
            probabilities, classes = logits, self.classes_
        flat = restrict_predictions(probabilities, classes, tasks)
        result, cursor = [], 0
        for ex in examples:
            size = 2*len(ex['words'])+1
            result.append(flat[cursor:cursor+size]); cursor += size
        return result

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(path/'tokenizer')
        self.network.encoder.config.to_json_file(path/'encoder_config.json')
        torch.save(self.network.state_dict(), path/'weights.pt')
        metadata = {k: v for k, v in self.__dict__.items() if k not in ['network', 'tokenizer', 'device']}
        with open(path/'metadata.pkl', 'wb') as stream:
            pickle.dump(metadata, stream)

    @classmethod
    def load(cls, path, device='cpu'):
        path = Path(path)
        # Load only artifacts produced by this tool: pickle is not a safe interchange format.
        with open(path/'metadata.pkl', 'rb') as stream:
            metadata = pickle.load(stream)
        model = cls.__new__(cls)
        model.__dict__.update(metadata)
        model.device = torch.device(device)
        model.tokenizer = AutoTokenizer.from_pretrained(path/'tokenizer', local_files_only=True)
        encoder = BertModel(BertConfig.from_json_file(path/'encoder_config.json'))
        model.network = RestorationBert(encoder, len(model.classes_), len(LINGUISTIC) if model.builder.condition == 'E' else 0).to(model.device)
        model.network.load_state_dict(torch.load(path/'weights.pt', map_location=model.device, weights_only=True))
        model.network.eval()
        return model
