"""One-to-one exact/relaxed span scoring after word alignment."""
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .alignment import align
from .data import LABELS


def spans(tags):
    result = []
    current = None
    for i, tag in enumerate(list(tags) + ['O']):
        if tag == 'O':
            prefix, label = 'O', None
        else:
            prefix, label = tag.split('-', 1)
        if current and (label != current[0] or prefix != 'I'):
            result.append((current[0], current[1], i))
            current = None
        if label and current is None:
            current = (label, i)
    return result


def prf(tp, predicted, gold):
    p, r = tp/predicted if predicted else 0., tp/gold if gold else 0.
    return {'precision': p, 'recall': r, 'f1': 2*p*r/(p+r) if p+r else 0.,
            'tp': int(tp), 'predicted': int(predicted), 'support': int(gold)}


def score_restorations(gold, predicted, include_restoration=True):
    if len(gold) != len(predicted):
        raise ValueError('Gold/prediction example counts differ')
    counts = {mode: defaultdict(lambda: [0, 0, 0]) for mode in ['exact', 'relaxed']}
    for mode in counts:
        for label in LABELS:
            counts[mode][label] = [0, 0, 0]
    edits, denominator = 0, 0
    confusion = Counter()
    for reference, hypothesis in zip(gold, predicted):
        if reference['id'] != hypothesis['id']:
            raise ValueError('Gold/prediction ids differ')
        gs, ps = spans(reference['tags']), spans(hypothesis['tags'])
        alignment = align(reference['words'], hypothesis['words'])
        matching = {e.observed: e.expected for e in alignment if e.op == 'M'}
        tag_alignment = align(list(zip(reference['words'], reference['tags'])),
                              list(zip(hypothesis['words'], hypothesis['tags'])))
        edits += sum(e.op != 'M' for e in tag_alignment)
        denominator += len(reference['words'])
        for e in alignment:
            gt = reference['tags'][e.expected] if e.expected is not None and e.op != 'I' else '<inserted>'
            pt = hypothesis['tags'][e.observed] if e.observed is not None else '<deleted>'
            confusion[(gt, pt)] += 1
        labels = {s[0] for s in gs + ps}
        for label in labels:
            g = [s for s in gs if s[0] == label]
            p = [s for s in ps if s[0] == label]
            for mode in counts:
                valid = np.zeros((len(g), len(p)), dtype=int)
                for i, (_, a, b) in enumerate(g):
                    for j, (_, c, d) in enumerate(p):
                        mapped = [matching.get(k) for k in range(c, d)]
                        if mode == 'exact':
                            valid[i, j] = mapped == list(range(a, b))
                        else:
                            valid[i, j] = any(k is not None and a <= k < b for k in mapped)
                matched = 0
                if valid.size:
                    ri, ci = linear_sum_assignment(-valid)
                    matched = int(valid[ri, ci].sum())
                c = counts[mode][label]
                c[0] += matched
                c[1] += len(p)
                c[2] += len(g)
    result = {}
    for mode, by_label in counts.items():
        per = {label: prf(*c) for label, c in sorted(by_label.items())}
        totals = [sum(c[i] for c in by_label.values()) for i in range(3)]
        present = [v['f1'] for v in per.values() if v['support']]
        result[mode] = {'per_tag': per, 'micro': prf(*totals),
                        'macro_f1': float(np.mean(present)) if present else 0.}
    result['tag_aware_wer'] = edits/max(1, denominator)
    result['confusion'] = [{'gold': g, 'predicted': p, 'count': n} for (g, p), n in sorted(confusion.items())]
    if include_restoration and all('inserted' in r for r in gold+predicted):
        def only_insertions(records):
            return [dict(r, tags=[tag if ins else 'O' for tag, ins in zip(r['tags'], r['inserted'])])
                    for r in records]
        result['restoration_only'] = score_restorations(only_insertions(gold), only_insertions(predicted), False)
    return result
