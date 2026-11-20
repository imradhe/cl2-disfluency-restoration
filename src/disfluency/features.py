"""Fold-fitted surprisal and duration, and paper-defined DLT costs."""
import math
from collections import Counter, defaultdict

from .alignment import phone_evidence

LINGUISTIC = ['left_surprisal', 'right_surprisal', 'left_duration', 'right_duration',
              'left_elongation', 'right_elongation', 'left_syllable_duration', 'right_syllable_duration',
              'left_ic', 'right_ic', 'left_sc', 'right_sc', 'left_synsurp', 'right_synsurp',
              'left_synsurp_available', 'right_synsurp_available']
VOWELS = set('aa ae ah ao aw ax ay eh er ey ih iy ow oy uh uw'.split())


def dlt(heads):
    """Dammalapati et al. §2.2: minimum 1; lengths count intervening words.

    Storage counts already-seen dependents whose HEAD is still to come; it
    does not count unseen dependents of an already-seen head. Roots use -1.
    """
    n = len(heads)
    if any(not isinstance(h, int) or h < -1 or h >= n or h == i for i, h in enumerate(heads)):
        raise ValueError('heads must contain zero-based heads, with root=-1')
    for i in range(n):
        seen, j = set(), i
        while j != -1:
            if j in seen:
                raise ValueError('Dependency cycle')
            seen.add(j)
            j = heads[j]
    ic = [1.] * n
    for dep, head in enumerate(heads):
        if head != -1:
            ic[max(dep, head)] += abs(dep-head)-1
    sc = [1. + sum(h > i for h in heads[:i+1]) for i in range(n)]
    return ic, sc


def add_dependencies(example, nlp):
    """Parse already-tokenized observed words; never parse the disfluent reference."""
    from spacy.tokens import Doc
    doc = Doc(nlp.vocab, words=[w['text'] for w in example['words']])
    for _, component in nlp.pipeline:
        doc = component(doc)
    example = dict(example)
    example['heads'] = [-1 if t.head.i == t.i else t.head.i for t in doc]
    return example


class FeatureBuilder:
    def __init__(self, condition='D', strict_syntax=True):
        if condition not in 'ABCDE' or len(condition) != 1:
            raise ValueError('Use ablation A, B, C, D or E')
        self.condition = condition
        self.strict_syntax = strict_syntax

    def fit(self, examples):
        self.trigrams, self.contexts = Counter(), Counter()
        self.vocab = {'<unk>', '</s>'}
        self.duration_sum, self.duration_n = Counter(), Counter()
        # Called exclusively on the training fold, before any sampling.
        for ex in examples:
            words = [w['text'].lower() for w in ex['words']]
            self.vocab.update(words)
            history = ['<s>', '<s>']
            for word in words + ['</s>']:
                ctx = tuple(history[-2:])
                self.contexts[ctx] += 1
                self.trigrams[ctx + (word,)] += 1
                history.append(word)
            for w in ex['words']:
                if w.get('duration') is not None:
                    self.duration_sum[w['text']] += w['duration']
                    self.duration_n[w['text']] += 1
        self.mean_duration = sum(self.duration_sum.values()) / max(1, sum(self.duration_n.values()))
        return self

    def linguistic(self, ex):
        n = len(ex['words'])
        if self.condition == 'E' and self.strict_syntax and 'heads' not in ex:
            raise ValueError('Condition E requires dependency heads or --parser-model; no locality proxy is substituted')
        if 'heads' in ex:
            if len(ex['heads']) != n:
                raise ValueError('Dependency heads must match the observed transcript words')
            ic, sc = dlt(ex['heads'])
        else:
            ic, sc = [0.] * n, [0.] * n
        history, result = ['<s>', '<s>'], []
        syn = ex.get('syntactic_surprisal', [None] * n)
        if len(syn) != n:
            raise ValueError('Syntactic surprisal must have one value per observed word')
        for i, word in enumerate(ex['words']):
            text = word['text'].lower()
            token = text if text in self.vocab else '<unk>'
            ctx = tuple(history[-2:])
            probability = (self.trigrams[ctx + (token,)] + .1) / (self.contexts[ctx] + .1*len(self.vocab))
            history.append(token)
            duration = word.get('duration')
            if self.condition == 'E' and duration is None:
                raise ValueError('Condition E requires aligned word durations')
            duration = duration or 0.
            avg = self.duration_sum[text]/self.duration_n[text] if self.duration_n[text] else self.mean_duration
            syllables = max(1, sum(p.rstrip('012') in VOWELS for p in word['canonical']))
            result.append(dict(surprisal=-math.log(probability), duration=duration,
                               elongation=duration/max(avg, 1e-6), syllable_duration=duration/syllables,
                               ic=ic[i], sc=sc[i], synsurp=float(syn[i] or 0.)))
        return result

    def rows(self, ex):
        words, n = ex['words'], len(ex['words'])
        evidence = phone_evidence([w['canonical'] for w in words], ex['observed_phones']) if self.condition in 'DE' else None
        ling = self.linguistic(ex) if self.condition == 'E' else None
        rows = []
        for task, indices in [('word', range(n)), ('gap', range(n+1))]:
            for i in indices:
                f = {'task': task}
                for delta in [-1, 0, 1]:
                    j = i+delta
                    f[f'word{delta}'] = words[j]['text'] if 0 <= j < n else ('<bos>' if j < 0 else '<eos>')
                if self.condition == 'B':
                    f['canonical'] = ' '.join(words[i]['canonical']) if i < n else ''
                if self.condition == 'C':
                    # Raw observed stream, without canonical-based allocation (clean ablation).
                    for phone in ex['observed_phones']:
                        f['observed:' + phone] = f.get('observed:' + phone, 0) + 1
                if evidence is not None:
                    f['gap_phones'] = ' '.join(evidence['gaps'][i])
                    f['gap_phone_count'] = len(evidence['gaps'][i])
                    if i < n:
                        w = evidence['words'][i]
                        f.update({key: w[key] for key in ['matches', 'substitutions', 'deletions', 'insertions']})
                        f.update(canonical=' '.join(words[i]['canonical']), realized=' '.join(w['realized']), ops=''.join(w['ops']))
                    # Timing is held out until E, keeping the feature addition explicit.
                if ling is not None:
                    for prefix, j in [('left', i-1), ('right', i)]:
                        for key in ['surprisal', 'duration', 'elongation', 'syllable_duration', 'ic', 'sc', 'synsurp']:
                            f[f'{prefix}_{key}'] = ling[j][key] if 0 <= j < n else 0.
                        f[f'{prefix}_synsurp_available'] = int('syntactic_surprisal' in ex and 0 <= j < n and ex['syntactic_surprisal'][j] is not None)
                rows.append(f)
        return rows

    def serialize(self, ex, i, task, evidence=None):
        """Readable word | /canonical/ | /realized/ | ops | gap serialization."""
        words, n = ex['words'], len(ex['words'])
        if i == n:
            text = '[END]'
        else:
            text = words[i]['text']
        if self.condition == 'A':
            return text
        if self.condition == 'B':
            return text + ' | /' + ' '.join(words[i]['canonical'] if i < n else []) + '/'
        if self.condition == 'C':
            return text  # raw stream appended once by the BERT encoder
        ev = evidence if evidence is not None else phone_evidence([w['canonical'] for w in words], ex['observed_phones'])
        w = ev['words'][i] if i < n else {'realized': [], 'ops': []}
        canonical = words[i]['canonical'] if i < n else []
        return (text + ' | /' + ' '.join(canonical) + '/ | /' + ' '.join(w['realized']) +
                '/ | ' + ''.join(w['ops']) + ' | gap /' + ' '.join(ev['gaps'][i]) + '/')
