#!/usr/bin/env python3
"""Export a verified, matched-feature full-corpus classifier comparison."""
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMES = {'logistic':'One-vs-rest logistic', 'multinomial':'Multinomial logistic',
         'svm':'Linear SVM (hinge)', 'mlp':'MLP'}


def compare(root=ROOT):
    root = Path(root)
    sources = {kind:root/'results/classification'/('glove50' if kind=='mlp' else 'glove50_'+kind)
               for kind in NAMES}
    if (root/'results/classification/glove50_glmm/summary.json').exists():
        sources['glmm'] = root/'results/classification/glove50_glmm'
    names = {**NAMES,'glmm':'Variational GLMM'}
    reports = {kind:json.loads((path/'summary.json').read_text()) for kind,path in sources.items()}
    reference = reports['mlp']['settings']
    expected = json.loads((root/'data/processed/label_stats.json').read_text())['labels']
    expected = {k:v for k,v in expected.items() if k!='NOISE'}
    total = sum(expected.values())
    rows, classes, records = [], [], {}
    for kind,report in reports.items():
        assert report['complete'], f'{kind}: incomplete cross-validation'
        settings = report['settings']
        assert settings.get('classifier','mlp')==kind
        for field in ['source_sha256','groups','vectors_sha256','embedding','epochs','batch_size',
                      'balancing','seed','dimension','min_frequency','hidden','head_lr','encoder_lr','tasks']:
            assert settings[field]==reference[field], f'{kind}: mismatched {field}'
        folds = [json.loads(path.read_text()) for path in sources[kind].glob('*fold*.json')]
        assert len(folds)==10*len(settings['tasks'])*len(settings['modes'])
        records[kind] = folds
        for mode in settings['modes']:
            view = 'static_'+mode
            def task_row(task):
                candidates = [r for r in report['summary'] if r['view']==view and r['task']==task]
                assert len(candidates)==1
                result = candidates[0]
                assert result['folds']==10 and result['test_examples_total']==total
                return result
            binary,multi = task_row('binary'),task_row('multiclass')
            b,m = binary['natural_prevalence'],multi['natural_prevalence']
            assert {name:score['support'] for name,score in m['per_class'].items()}==expected
            positive = b['per_class']['disfluent']
            rows.append({'classifier':kind,'embedding_mode':mode,'examples':total,'folds':10,
                         'binary_balanced_accuracy':binary['balanced_test']['accuracy']['mean'],
                         'binary_balanced_accuracy_std':binary['balanced_test']['accuracy']['std'],
                         'binary_natural_accuracy':b['accuracy']['mean'],
                         'binary_positive_precision':positive['precision_mean'],
                         'binary_positive_recall':positive['recall_mean'],
                         'binary_positive_f1':positive['f1_mean'],
                         'binary_positive_f1_std':positive['f1_std'],
                         'multiclass_accuracy':m['accuracy']['mean'],
                         'multiclass_accuracy_std':m['accuracy']['std'],
                         'multiclass_macro_f1':m['macro_f1']['mean'],
                         'multiclass_macro_f1_std':m['macro_f1']['std']})
            for name,score in m['per_class'].items():
                classes.append({'classifier':kind,'embedding_mode':mode,'class':name,**score})
            for task in ['binary','multiclass']:
                part = [r for r in folds if r['view']==view and r['task']==task]
                assert sorted(r['fold'] for r in part)==list(range(10))
                for record in part:
                    audit = record['training']
                    assert len(audit['history'])==reference['epochs']
                    assert all(e['examples_seen']==record['train_examples'] for e in audit['history'])
                    assert audit['encoder_changed']==(mode=='updated')
                    assert audit['embedding_weights_changed']==(mode=='updated')
    for fold in range(10):
        for task in ['binary','multiclass']:
            encoder_initializations = set()
            for kind,folds in records.items():
                pair = [r for r in folds if r['fold']==fold and r['task']==task]
                assert len(pair)==len(reports[kind]['settings']['modes'])
                assert len({r['training']['head_sha256_before'] for r in pair})==1
                encoder_initializations.update(r['training']['encoder_sha256_before'] for r in pair)
            assert len(encoder_initializations)==1, 'Different embedding initializations'
    baseline_settings = json.loads((root/'results/classification/full/settings.json').read_text())
    assert reference['groups']==baseline_settings['groups']
    directory = root/'results/classification/model_comparison'
    directory.mkdir(parents=True,exist_ok=True)
    for filename,table in [('comparison.csv',rows),('per_class.csv',classes)]:
        with (directory/filename).open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(table)
    verification = {'examples':total,'conversations':len(reference['groups']),'folds':10,
                    'fold_fits':sum(len(folds) for folds in records.values()),
                    'same_source':True,'same_gloVe_vectors':True,'same_conversation_folds':True,
                    'same_embedding_initializations':True,'matched_initial_heads_within_mode_pairs':True,
                    'every_training_row_used_each_epoch':True,'per_class_supports_match_source':True,
                    'frozen_weights_unchanged':True,'updated_weights_changed':True,
                    'source_sha256':reference['source_sha256'],'vectors_sha256':reference['vectors_sha256'],
                    'epochs':reference['epochs'],'transcript_only':True,'sample_limit':None}
    (directory/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    text = '''# Comparison across modelling techniques

All **3,085,832 lexical examples** from **2,438 conversations** are evaluated in the same ten
conversation-separated folds. Each technique uses the same 50-dimensional pretrained GloVe
initialization and four positional context words (200 input coordinates). Target words, timings,
phones and audio are excluded. Candidate contexts are constructed from the existing annotations;
these are contextual classification results, not end-to-end restoration.

All models use training-only inverse-frequency sample weights, two complete optimization epochs,
batches of 8,192 and AdamW with learning rate 0.001 and weight decay 0.01. Linear heads have no
hidden layer; MLP has one 64-unit ReLU layer and 0.1 dropout. This is a **fixed-budget comparison**,
not a claim that every optimizer has converged or that hyperparameters are optimal. No outer-test
fold is used for epoch/hyperparameter selection.

One-vs-rest logistic uses independent sigmoid binary cross-entropy; multinomial logistic uses
joint softmax cross-entropy; linear SVM uses one-vs-rest hinge loss. MLP uses softmax cross-entropy.
All are trained with minibatches in PyTorch. The SVM is a stochastic hinge-loss model, not a liblinear
fit. Class weighting applies to training rows for each model; evaluation retains natural prevalence.

Percentages below are means across ten held-out folds. Binary accuracy is evaluated on the same
fixed equal-class held-out subsets. Binary positive F1 and all multiclass metrics use the complete
natural-prevalence test data. Standard deviations, precision/recall and counts are retained in CSV.
'''
    for mode,title in [('frozen','Frozen GloVe: classifier comparison'),('updated','Task-updated GloVe: joint representation/classifier training')]:
        text+=f'\n## {title}\n\n'
        text+='| Technique | Binary balanced accuracy | Binary positive F1 | Ten-class accuracy | Ten-class macro F1 |\n'
        text+='|---|---:|---:|---:|---:|\n'
        for row in rows:
            if row['embedding_mode']!=mode:
                continue
            text+='| '+names[row['classifier']]+' | '+' | '.join(f"{row[field]*100:.2f}%" for field in
                   ['binary_balanced_accuracy','binary_positive_f1','multiclass_accuracy','multiclass_macro_f1'])+' |\n'
        active = [kind for kind in sources if any(r['classifier']==kind and r['embedding_mode']==mode for r in rows)]
        text+='\nTen-class F1 by label:\n\n| Class | '+' | '.join(names[kind] for kind in active)+' |\n|---|'+'---:|'*len(active)+'\n'
        for name in expected:
            values = [next(r for r in classes if r['classifier']==kind and r['embedding_mode']==mode and r['class']==name)['f1_mean']
                      for kind in active]
            text+='| '+name+' | '+' | '.join(f'{value*100:.2f}%' for value in values)+' |\n'
    text+='''
## Other baselines and missing experiments

The hashed-feature SGD logistic baseline obtained 76.27% binary balanced accuracy, 50.78% binary
positive F1, 82.64% ten-class accuracy and 24.86% ten-class macro F1. It uses a different feature
representation and is therefore a separate baseline, not a matched-GloVe classifier comparison.

The GLMM row, when present, uses a factorized Gaussian variational approximation for a logistic
model with Gaussian random intercepts by preceding context word. Infrequent training words share
a rare-word group; unseen groups use the population prior. Fixed effects use the same frozen GloVe
coordinates. The expected log likelihood uses 12-point Gauss-Hermite quadrature and a Gaussian KL
penalty; this is not the paper's lme4 estimator or the restoration pilot's direct marginal likelihood
optimizer. The two-epoch budget does not establish variational convergence. Posterior diagnostics
are retained in each fold report.

Full-corpus BERT progress, completion state and resume command are in
[the BERT run record](../bert_full/README.md). Small restoration integration runs
do not belong in these tables because their data size, target and metrics differ. No pilot result
is presented as full-corpus evidence. Other unrun pretrained embedding families are also omitted.

## Files and reproduction

- [All comparison metrics](comparison.csv)
- [Per-class precision, recall, F1 and support](per_class.csv)
- [Matched-data/fold/initialization verification](verification.json)
- [MLP fold results and final model paths](../glove50/README.md)
- [OvR logistic report](../glove50_logistic/summary.json)
- [Multinomial logistic report](../glove50_multinomial/summary.json)
- [Linear SVM report](../glove50_svm/summary.json)
- [Variational GLMM report](../glove50_glmm/summary.json)

For each `CLASSIFIER` in `logistic multinomial svm mlp`, run:

```bash
.venv/bin/python scripts/classify_embeddings.py \\
  --embedding static --classifier CLASSIFIER \\
  --vectors .cache/vectors/glove-wiki-gigaword-50.txt \\
  --tasks binary multiclass --fold-manifest results/classification/full/settings.json \\
  --epochs 2 --batch-size 8192 --hidden 64 \\
  --out results/classification/glove50_CLASSIFIER --resume \\
  --fit-final --checkpoints ckpt/classification_glove50_CLASSIFIER
```

The existing MLP report is `glove50/` (omit `_mlp` in its output path). Final-model refits use all
eligible rows and are saved locally under ignored `ckpt/`; their training data are not scored as
held-out evaluation. Run `python scripts/compare_classifiers.py` to validate and recreate these tables.
'''
    (directory/'README.md').write_text(text)
    return rows


if __name__=='__main__':
    rows = compare()
    for row in rows:
        print(row['classifier'],row['embedding_mode'],
              f"binary balanced ACC {row['binary_balanced_accuracy']*100:.2f}%, "
              f"multiclass macro F1 {row['multiclass_macro_f1']*100:.2f}%")
