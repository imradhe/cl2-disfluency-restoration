"""Shared conversation-held-out folds, model persistence, and ablation reports."""
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from .baselines import ClassicalTagger, rule_predictions
from .data import LABELS, decode, targets, validate, write_jsonl
from .features import FeatureBuilder
from .metrics import score_restorations


def folds(examples, n_splits=10, seed=42, manifest=None):
    groups = Counter(str(e['conversation']) for e in examples)
    if len(groups) < n_splits:
        raise ValueError(f'{n_splits} folds require at least {n_splits} conversations; found {len(groups)}')
    ids = [e['id'] for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError('Example ids must be unique (give gold/ASR variants different ids and a shared pair_id)')
    fingerprint = hashlib.sha256(json.dumps(examples, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    manifest = Path(manifest) if manifest else None
    if manifest and manifest.exists():
        result = json.loads(manifest.read_text())
        if result['fingerprint'] != fingerprint or result['n_splits'] != n_splits or result['seed'] != seed:
            raise ValueError('Fold manifest belongs to different data, fold count or seed; choose a new path')
        return result
    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    # Greedy allocation balances example counts while keeping both conversation sides together.
    shuffled.sort(key=lambda group: -groups[group])
    load, assigned = [0]*n_splits, {}
    for group in shuffled:
        fold = int(np.argmin(load))
        assigned[group] = fold
        load[fold] += groups[group]
    result = {'fingerprint': fingerprint, 'seed': seed, 'n_splits': n_splits,
              'grouping': 'conversation (both speaker sides together)', 'groups': assigned, 'fold_sizes': load}
    if manifest:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(result, indent=2)+'\n')
    return result


def make_model(kind, condition, balancing, args, representation='tfidf'):
    builder = FeatureBuilder(condition)
    if kind == 'bert':
        from .bert import BertTagger
        return BertTagger(builder, encoder=args.encoder, cache_dir=str(args.cache_dir),
                          tiny=args.tiny, local_only=not args.allow_download, epochs=args.epochs,
                          batch_size=args.batch_size, learning_rate=args.learning_rate,
                          balancing=balancing, seed=args.seed, max_length=args.max_length, device=args.device)
    return ClassicalTagger(builder, model=kind, balancing=balancing, seed=args.seed,
                           representation=representation, vectors=args.vectors, encoder=args.encoder,
                           cache_dir=args.cache_dir, local_only=not args.allow_download,
                           batch_size=args.batch_size, device=args.device)


def save_model(model, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, 'network'):
        model.save(path)
        kind = 'bert'
    else:
        with open(path/'model.pkl', 'wb') as f:
            pickle.dump(model, f)
        kind = 'classical'
    (path/'kind.json').write_text(json.dumps({'kind': kind})+'\n')


def load_model(path, device='cpu'):
    path = Path(path)
    if json.loads((path/'kind.json').read_text())['kind'] == 'bert':
        from .bert import BertTagger
        return BertTagger.load(path, device)
    with open(path/'model.pkl', 'rb') as f:
        return pickle.load(f)


def cross_validate(examples, args):
    for ex in examples:
        validate(ex, training=True)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    manifest = folds(examples, args.folds, args.seed, args.fold_manifest or output/'folds.json')
    conditions = list(args.conditions)
    if 'E' in conditions or ('F' in conditions and args.f_base == 'E'):
        if any('heads' not in ex or any(w.get('duration') is None for w in ex['words']) for ex in examples):
            raise ValueError('E requires dependency heads and timed-transcript durations for every example')
    if 'F' in conditions:
        pairs = {view: {e.get('pair_id') for e in examples if e.get('word_origin') == view} for view in ['gold', 'asr']}
        if not pairs['gold'] or pairs['gold'] != pairs['asr'] or None in pairs['gold']:
            raise ValueError('F requires paired supplied gold/asr transcripts with matching pair_id; no ASR audio stage is run')
        pair_groups = {}
        for ex in examples:
            if ex.get('word_origin') in pairs:
                pair_groups.setdefault(ex['pair_id'], set()).add(str(ex['conversation']))
        if any(len(g) != 1 for g in pair_groups.values()):
            raise ValueError('Paired transcripts must share their conversation group')
        conditions = [c for c in conditions if c != 'F'] + ['F_gold', 'F_asr']
    records = []
    provenance = {field: sorted({e.get(field, 'unspecified') for e in examples})
                  for field in ['phone_source', 'label_source', 'word_origin', 'parse_source']}
    for condition in conditions:
        for kind in args.models:
            representations = (['finetuned'] if kind == 'bert' else ['numeric'] if kind == 'glmm' else ['rules'] if kind == 'rule' else args.representations)
            for representation in representations:
                for balancing in (['none'] if kind == 'rule' else args.balancing):
                    for fold in range(args.folds):
                        base = args.f_base if condition.startswith('F_') else condition
                        train = [e for e in examples if manifest['groups'][str(e['conversation'])] != fold
                                 and e.get('word_origin') != 'asr']
                        view = condition.split('_', 1)[1] if condition.startswith('F_') else None
                        if view:
                            train = [e for e in train if e.get('word_origin') == 'gold']
                        test = [e for e in examples if manifest['groups'][str(e['conversation'])] == fold
                                and (e.get('word_origin') == view if view else e.get('word_origin') != 'asr')]
                        if not train or not test:
                            raise ValueError(f'Empty train/test partition: {condition} fold {fold}')
                        assert not ({str(e['conversation']) for e in train} & {str(e['conversation']) for e in test})
                        name = f'{condition}_{kind}_{representation}_{balancing}_fold{fold:02d}'
                        print(name, f'train={len(train)} test={len(test)}', flush=True)
                        if kind == 'rule':
                            predicted, sampling = rule_predictions(test), {'policy': 'none'}
                        else:
                            model = make_model(kind, base, balancing, args, representation).fit(train)
                            predicted, sampling = model.predict(test), model.sampling_report
                            if args.save_models:
                                save_model(model, Path(args.checkpoints)/name)
                        gold = [decode(ex, targets(ex)) for ex in test]
                        hypothesis = [decode(ex, pred) for ex, pred in zip(test, predicted)]
                        metrics = score_restorations(gold, hypothesis)
                        record = {'condition': condition, 'model': kind, 'representation': representation, 'balancing': balancing, 'fold': fold,
                                  'train_examples': len(train), 'test_examples': len(test), 'metrics': metrics,
                                  'sampling': sampling, 'provenance': provenance}
                        records.append(record)
                        (output/(name+'_metrics.json')).write_text(json.dumps(record, indent=2)+'\n')
                        write_jsonl(output/(name+'_predictions.jsonl'), hypothesis)
                        # Preserve partial progress after each successful fold.
                        (output/'fold_results.json').write_text(json.dumps(records, indent=2)+'\n')
    return write_summary(output, records, manifest, provenance)


def write_summary(output, records, manifest, provenance):
    output = Path(output)
    summary = []
    keys = sorted({(r['condition'], r['model'], r['representation'], r['balancing']) for r in records})
    for condition, kind, representation, balancing in keys:
        selected = [r for r in records if (r['condition'], r['model'], r['representation'], r['balancing']) == (condition, kind, representation, balancing)]
        row = {'condition': condition, 'model': kind, 'representation': representation, 'balancing': balancing, 'folds': len(selected)}
        paths = [('exact_micro_f1', lambda m: m['exact']['micro']['f1']),
                 ('relaxed_micro_f1', lambda m: m['relaxed']['micro']['f1']),
                 ('exact_macro_f1', lambda m: m['exact']['macro_f1']),
                 ('restoration_exact_f1', lambda m: m['restoration_only']['exact']['micro']['f1']),
                 ('restoration_relaxed_f1', lambda m: m['restoration_only']['relaxed']['micro']['f1']),
                 ('tag_aware_wer', lambda m: m['tag_aware_wer'])]
        for metric, fn in paths:
            values = [fn(r['metrics']) for r in selected]
            row[metric+'_mean'] = float(np.mean(values))
            row[metric+'_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        row['per_tag'] = {}
        tags = sorted(set(LABELS) | {t for r in selected for t in r['metrics']['exact']['per_tag']})
        for tag in tags:
            row['per_tag'][tag] = {}
            for mode in ['exact', 'relaxed']:
                for metric in ['precision', 'recall', 'f1']:
                    values = [r['metrics'][mode]['per_tag'].get(tag, {}).get(metric, 0.) for r in selected]
                    row['per_tag'][tag][mode+'_'+metric+'_mean'] = float(np.mean(values))
                    row['per_tag'][tag][mode+'_'+metric+'_std'] = float(np.std(values, ddof=1)) if len(values)>1 else 0.
        summary.append(row)
    report = {'protocol': manifest, 'provenance': provenance, 'summary': summary,
              'notes': ['Dictionary-derived phone streams are simulations, not audio recognition.',
                        'MS98 project-rule labels are not NXT/Treebank gold labels.',
                        'The local add-0.1 trigram LM differs from the paper OANC/SRILM LM.',
                        'Syntactic surprisal is optional supplied data; missing values have availability flags.',
                        'GLMM uses Gaussian-Hermite marginal likelihood, not the paper lme4 estimator.']}
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    import pandas as pd
    pd.DataFrame([{k:v for k,v in r.items() if k != 'per_tag'} for r in summary]).to_csv(output/'summary.csv', index=False)
    return report
