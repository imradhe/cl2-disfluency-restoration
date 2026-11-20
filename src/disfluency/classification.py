"""Full-corpus context-based binary and class-wise disfluency prediction.

Each lexical token is an example. Its own word/phones/label are withheld from
features. Contexts use the nearest fluent (O) words on either side within the
utterance, as a paper-inspired prediction task. This is not missing-word text
generation and is distinct from the small restoration-template pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from itertools import groupby
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import LABELS, read_jsonl

CLASSES = ['O'] + [label for label in LABELS if label != 'NOISE']
FILLERS = {'FILLER_HESITATION', 'FILLER_BACKCHANNEL', 'FILLER_DM'}
REPARANDA = {'REPETITION', 'REPAIR', 'FALSE_START', 'PARTIAL_WORD'}
HASH_SIZE = 2**18
CACHE_VERSION = 1


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(processed, directory):
    """Read every side once; compact arrays rather than millions of Python dicts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = file_hash(processed)
    manifest_path = directory/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('source_sha256') == fingerprint and manifest.get('cache_version') == CACHE_VERSION:
            return load_cache(directory)
        raise ValueError('Classification cache differs from source/version; use a new cache directory')
    vocabulary = {'<bos>': 0, '<eos>': 1}
    word_list = ['<bos>', '<eos>']
    label_id = {label: i for i, label in enumerate(CLASSES)}
    chunks = {key: [] for key in ['contexts', 'durations', 'labels', 'groups', 'speaker_sides']}
    counts = Counter()
    tokens_read, noise_skipped, sides = 0, 0, 0
    conversations = set()
    started = time.monotonic()
    for side_index, side in enumerate(read_jsonl(processed)):
        sides += 1
        conversation = int(side['conv'])
        conversations.add(conversation)
        side_contexts, side_durations, side_labels = [], [], []
        for _, group in groupby(side['tokens'], key=lambda token: token['utt']):
            source = list(group)
            tokens_read += len(source)
            tokens = [t for t in source if t['kind'] != 'NOISE' and t['label'] != 'NOISE']
            noise_skipped += len(source)-len(tokens)
            if not tokens:
                continue
            ids = []
            for token in tokens:
                word = (token['spoken'] or token['raw']).lower()
                if word not in vocabulary:
                    vocabulary[word] = len(word_list)
                    word_list.append(word)
                ids.append(vocabulary[word])
            fluent = np.array([i for i,t in enumerate(tokens) if t['label'] == 'O'], dtype=np.int32)
            for i, token in enumerate(tokens):
                # Strictly before/after the target. Even a fluent target's own
                # word and duration are excluded, keeping conditions comparable.
                left_count = int(np.searchsorted(fluent, i, side='left'))
                right_start = int(np.searchsorted(fluent, i, side='right'))
                left = int(fluent[left_count-1]) if left_count else None
                left2 = int(fluent[left_count-2]) if left_count > 1 else None
                right = int(fluent[right_start]) if right_start < len(fluent) else None
                right2 = int(fluent[right_start+1]) if right_start+1 < len(fluent) else None
                context = [ids[j] if j is not None else (0 if k < 2 else 1)
                           for k,j in enumerate([left2,left,right,right2])]
                durations = []
                for j in [left,right]:
                    if j is None or tokens[j].get('start') is None or tokens[j].get('end') is None:
                        durations.append(-1.)
                    else:
                        durations.append(max(0.,tokens[j]['end']-tokens[j]['start']))
                label = token['label']
                side_contexts.append(context)
                side_durations.append(durations)
                side_labels.append(label_id[label])
                counts[label] += 1
        n = len(side_labels)
        if n:
            chunks['contexts'].append(np.asarray(side_contexts,dtype=np.int32))
            chunks['durations'].append(np.asarray(side_durations,dtype=np.float32))
            chunks['labels'].append(np.asarray(side_labels,dtype=np.uint8))
            chunks['groups'].append(np.full(n,conversation,dtype=np.int32))
            chunks['speaker_sides'].append(np.full(n,side_index,dtype=np.int16))
        if sides % 250 == 0:
            print(f'Prepared {sides:,} sides / {sum(counts.values()):,} lexical examples',flush=True)
    if not counts:
        raise ValueError('No lexical examples in processed corpus')
    for key, parts in chunks.items():
        np.save(directory/(key+'.npy'),np.concatenate(parts))
    (directory/'vocabulary.json').write_text(json.dumps(word_list,ensure_ascii=False)+'\n')
    manifest = {'cache_version':CACHE_VERSION,'source':str(processed),'source_sha256':fingerprint,
                'source_token_rows':tokens_read,'noise_rows_excluded':noise_skipped,
                'examples':sum(counts.values()),'sides':sides,'conversations':len(conversations),
                'class_counts':dict(counts),'classes':CLASSES,
                'label_source':'existing MS98 project-rule labels, including silver categories',
                'context':'nearest two fluent words each side, same utterance, target hidden',
                'audio_used':False,'sample_limit':None,
                'preparation_seconds':time.monotonic()-started}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return load_cache(directory)


def load_cache(directory):
    directory = Path(directory)
    arrays = {key:np.load(directory/(key+'.npy'),mmap_mode='r')
              for key in ['contexts','durations','labels','groups','speaker_sides']}
    return arrays,json.loads((directory/'manifest.json').read_text())


def context_matrix(contexts):
    """Fixed hashing (no vocabulary fitting on held-out conversations)."""
    n = len(contexts)
    ids = np.asarray(contexts,dtype=np.int64)
    columns = [(ids[:,j]*2654435761 + j*97531) % HASH_SIZE for j in range(4)]
    columns += [((ids[:,0]*1000003)^ids[:,1]^572191) % HASH_SIZE,
                ((ids[:,1]*1000003)^ids[:,2]^936931) % HASH_SIZE,
                ((ids[:,2]*1000003)^ids[:,3]^171717) % HASH_SIZE]
    indices = np.column_stack(columns).astype(np.int32).ravel()
    indptr = np.arange(0,(n+1)*7,7,dtype=np.int32)
    x = sparse.csr_matrix((np.ones(n*7,dtype=np.float32),indices,indptr),shape=(n,HASH_SIZE))
    x.sum_duplicates()
    return x


def timing_matrix(durations):
    values = np.asarray(durations,dtype=np.float32)
    present = values >= 0
    values = np.maximum(values,0.)
    # Raw and log-duration of surrounding fluent words, with boundary/missing flags.
    return np.column_stack([values,np.log1p(values),present.astype(np.float32)])


def task_labels(labels, task):
    labels = np.asarray(labels)
    if task == 'multiclass':
        return np.ones(len(labels),bool),labels,list(CLASSES)
    if task == 'binary':
        return np.ones(len(labels),bool),(labels != 0).astype(np.uint8),['fluent','disfluent']
    positives = FILLERS if task == 'filler' else REPARANDA
    positive_ids = [CLASSES.index(label) for label in positives]
    positive = np.isin(labels,positive_ids)
    return (labels == 0)|positive,positive.astype(np.uint8),['fluent',task]


def metrics(y, predicted, names):
    ids = np.arange(len(names))
    p,r,f,s = precision_recall_fscore_support(y,predicted,labels=ids,zero_division=0)
    active = s > 0
    return {'accuracy':float(accuracy_score(y,predicted)),
            'macro_f1':float(f[active].mean()) if active.any() else 0.,
            'per_class':{name:{'precision':float(p[i]),'recall':float(r[i]),'f1':float(f[i]),'support':int(s[i])}
                         for i,name in enumerate(names)},
            'confusion':confusion_matrix(y,predicted,labels=ids).tolist(),
            'examples':len(y)}


def balanced_test_indices(y,seed):
    rng = np.random.default_rng(seed)
    labels,counts = np.unique(y,return_counts=True)
    size = int(min(counts))
    chosen = np.concatenate([rng.choice(np.flatnonzero(y == label),size,replace=False) for label in labels])
    return chosen


def run(args):
    threadpool_limits(limits=args.threads)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    arrays,manifest = prepare(args.processed,args.cache)
    labels,groups = arrays['labels'],arrays['groups']
    print(json.dumps(manifest,indent=2),flush=True)
    started = time.monotonic()
    x_context = context_matrix(arrays['contexts'])
    timings = timing_matrix(arrays['durations'])
    split = GroupKFold(n_splits=args.folds)
    fold_of_row = np.empty(len(labels),dtype=np.uint8)
    group_folds = {}
    for fold,(_,test) in enumerate(split.split(np.zeros(len(labels),dtype=np.uint8),labels,groups)):
        fold_of_row[test] = fold
        for group in np.unique(groups[test]):
            group_folds[str(group)] = fold
    settings = {'source_sha256':manifest['source_sha256'],'folds':args.folds,'seed':args.seed,
                'epochs':args.epochs,'alpha':args.alpha,'balancing':args.balancing,
                'views':args.views,'tasks':args.tasks,'groups':group_folds,
                'classifier':'averaged SGD logistic regression',
                'features':'hashed surrounding fluent-word identities/bigrams; optional surrounding durations',
                'target_word_used':False,'full_corpus':True}
    settings_path = out/'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Output directory has different settings; choose a new output directory')
    settings_path.write_text(json.dumps(settings,indent=2)+'\n')
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results = []
    for view in args.views:
        for fold in range(args.folds):
            train_fold = fold_of_row != fold
            if view == 'context_timing':
                scaler = StandardScaler().fit(timings[train_fold])
                x = sparse.hstack([x_context,sparse.csr_matrix(scaler.transform(timings))],format='csr',dtype=np.float32)
            else:
                x = x_context
            for task in args.tasks:
                result_path = out/f'{view}_{task}_fold{fold:02d}.json'
                if args.resume and result_path.exists():
                    results.append(json.loads(result_path.read_text()))
                    continue
                mask,y,names = task_labels(labels,task)
                train = np.flatnonzero(train_fold & mask)
                test = np.flatnonzero(~train_fold & mask)
                assert not np.intersect1d(np.unique(groups[train]),np.unique(groups[test])).size
                began = time.monotonic()
                counts = np.bincount(y[train],minlength=len(names))
                weights = None
                if args.balancing == 'balanced':
                    weights = {i:len(train)/(len(names)*max(1,int(count))) for i,count in enumerate(counts)}
                model = SGDClassifier(loss='log_loss',penalty='l2',alpha=args.alpha,
                                      max_iter=args.epochs,tol=None,average=True,
                                      random_state=args.seed+fold,class_weight=weights,
                                      shuffle=True,n_jobs=1)
                print(f'{view} / {task} / fold {fold+1}/{args.folds}: train {len(train):,}, test {len(test):,}',flush=True)
                model.fit(x[train],y[train])
                predicted = model.predict(x[test])
                record = {'view':view,'task':task,'fold':fold,'train_examples':len(train),
                          'train_class_counts':{name:int(counts[i]) for i,name in enumerate(names)},
                          'natural_prevalence':metrics(y[test],predicted,names),
                          'seconds':time.monotonic()-began}
                if task != 'multiclass':
                    selected = balanced_test_indices(y[test],args.seed+fold)
                    record['balanced_test'] = metrics(y[test][selected],predicted[selected],names)
                result_path.write_text(json.dumps(record,indent=2)+'\n')
                results.append(record)
                score = record['natural_prevalence']
                print(f"  accuracy {score['accuracy']:.4f}, macro F1 {score['macro_f1']:.4f}, {record['seconds']:.1f}s",flush=True)
            if view == 'context_timing':
                del x
    summary = []
    for view in args.views:
        for task in args.tasks:
            rows = [r for r in results if r['view'] == view and r['task'] == task]
            row = {'view':view,'task':task,'folds':len(rows),'test_examples_total':sum(r['natural_prevalence']['examples'] for r in rows)}
            for evaluation in ['natural_prevalence','balanced_test']:
                if evaluation not in rows[0]:
                    continue
                row[evaluation] = {}
                for field in ['accuracy','macro_f1']:
                    values = [r[evaluation][field] for r in rows]
                    row[evaluation][field] = {'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1))}
                row[evaluation]['per_class'] = {}
                for name in rows[0][evaluation]['per_class']:
                    part = {}
                    for field in ['precision','recall','f1']:
                        values = [r[evaluation]['per_class'][name][field] for r in rows]
                        part[field+'_mean'] = float(np.mean(values))
                        part[field+'_std'] = float(np.std(values,ddof=1))
                    part['support'] = sum(r[evaluation]['per_class'][name]['support'] for r in rows)
                    row[evaluation]['per_class'][name] = part
                pooled = np.sum([r[evaluation]['confusion'] for r in rows],axis=0)
                row[evaluation]['pooled_confusion'] = pooled.tolist()
            summary.append(row)
    report = {'data':manifest,'settings':settings,'summary':summary,'elapsed_seconds':time.monotonic()-started,
              'notes':['Every lexical token in the full processed corpus is used; no sample/window limit.',
                       'The target word and target duration are excluded from features.',
                       'Context construction uses existing fluent/disfluent annotations to define candidate contexts.',
                       'These are context-based classification scores, not exact missing-word restoration scores.',
                       'Filler/reparandum family tasks use project-label mappings, not NXT gold categories.',
                       'Balanced-test scores use fixed equal-class subsets of held-out predictions; training is never fitted on test examples.']}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    import pandas as pd
    table=[]
    for row in summary:
        for evaluation in ['natural_prevalence','balanced_test']:
            if evaluation not in row:
                continue
            part=row[evaluation]
            table.append({'view':row['view'],'task':row['task'],'evaluation':evaluation,
                          'accuracy_mean':part['accuracy']['mean'],'accuracy_std':part['accuracy']['std'],
                          'macro_f1_mean':part['macro_f1']['mean'],'macro_f1_std':part['macro_f1']['std']})
    pd.DataFrame(table).to_csv(out/'summary.csv',index=False)
    export_tables(report,out)
    print(f'Full-corpus classification completed: {out}',flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed',type=Path,default=Path('data/processed/swb_tokens.jsonl.gz'))
    parser.add_argument('--cache',type=Path,default=Path('data/classification/full'))
    parser.add_argument('--out',type=Path,default=Path('results/classification/full'))
    parser.add_argument('--tasks',nargs='+',choices=['binary','multiclass','filler','reparandum'],default=['binary','multiclass','filler','reparandum'])
    parser.add_argument('--views',nargs='+',choices=['context','context_timing'],default=['context','context_timing'])
    parser.add_argument('--folds',type=int,default=10)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--alpha',type=float,default=1e-6)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--balancing',choices=['none','balanced'],default='balanced')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if args.epochs < 1 or args.folds < 2:
        parser.error('Positive epochs and at least two folds are required')
    run(args)


def export_tables(report, directory):
    """Readable class-wise and majority-baseline comparisons from OOF counts."""
    import pandas as pd
    directory=Path(directory)
    class_rows,task_rows=[],[]
    for row in report['summary']:
        for evaluation in ['natural_prevalence','balanced_test']:
            if evaluation not in row:
                continue
            part=row[evaluation]
            names=list(part['per_class'])
            confusion=np.asarray(part['pooled_confusion'],dtype=np.int64)
            support=confusion.sum(axis=1)
            total=int(support.sum())
            # Class 0 is fluent for every task in this experiment.
            fluent_baseline=float(support[0]/total) if total else 0.
            task_record={'view':row['view'],'task':row['task'],'evaluation':evaluation,
                         'folds':row['folds'],'test_examples_total':total,
                         'accuracy_mean':part['accuracy']['mean'],'accuracy_std':part['accuracy']['std'],
                         'macro_f1_mean':part['macro_f1']['mean'],'macro_f1_std':part['macro_f1']['std'],
                         'all_fluent_accuracy':fluent_baseline}
            if row['task']!='multiclass':
                positive=part['per_class'][names[1]]
                task_record.update(positive_precision_mean=positive['precision_mean'],
                                   positive_recall_mean=positive['recall_mean'],
                                   positive_f1_mean=positive['f1_mean'],positive_f1_std=positive['f1_std'])
            task_rows.append(task_record)
            for i,name in enumerate(names):
                stats=part['per_class'][name]
                class_rows.append({'view':row['view'],'task':row['task'],'evaluation':evaluation,
                                   'class':name,**stats,'true_positive_total':int(confusion[i,i]),
                                   'predicted_total':int(confusion[:,i].sum())})
    pd.DataFrame(task_rows).to_csv(directory/'comparison.csv',index=False)
    pd.DataFrame(class_rows).to_csv(directory/'per_class.csv',index=False)
    return task_rows,class_rows
