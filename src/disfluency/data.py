"""Data contracts and an explicitly simulated MS98 restoration task."""
import gzip
import hashlib
import json
import random
import re
from collections import Counter
from itertools import groupby
from pathlib import Path

LABELS = ['FILLER_HESITATION', 'FILLER_BACKCHANNEL', 'FILLER_DM', 'REPETITION',
          'REPAIR', 'FALSE_START', 'PARTIAL_WORD', 'PROLONGATION', 'MISPRONUNCIATION', 'NOISE']
REMOVABLE = set(LABELS[:7])
BIO = ['O'] + [f'{p}-{label}' for label in LABELS for p in ('B', 'I')]
NONE = '<none>'


def read_jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', encoding='utf8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def slot_key(units, words=None, slot_index=None):
    # Relative copy targets generalize repetitions/repairs to words never seen
    # as literal restoration classes in the training fold.
    encoded = []
    for unit in units:
        item = dict(unit)
        if words is not None and unit['label'] in {'REPETITION', 'REPAIR', 'FALSE_START'}:
            candidates = [j for j, word in enumerate(words)
                          if word['text'] == unit['text'] and abs(j-slot_index) <= 4]
            if candidates:
                j = min(candidates, key=lambda j: (abs(j-slot_index), j))
                item = {'label': unit['label'], 'copy': j-slot_index}
        encoded.append(item)
    return json.dumps(encoded, ensure_ascii=False, separators=(',', ':')) if encoded else NONE


def slot_units(key):
    return [] if key == NONE else json.loads(key)


def targets(example):
    return ['word:' + label for label in example['word_labels']] + [
        'gap:' + slot_key(units, example['words'], i) for i, units in enumerate(example['slots'])]


def validate(example, training=False):
    if not isinstance(example.get('id'), str) or not example['id']:
        raise ValueError('Every example needs a nonempty string id')
    if 'conversation' not in example:
        raise ValueError(f"{example['id']}: conversation is required for leakage-safe folds")
    words = example['words']
    if not words:
        raise ValueError(f"{example['id']}: at least one observed word is required")
    for word in words:
        if not word.get('text'):
            raise ValueError('Empty word text')
        if not isinstance(word.get('canonical'), list):
            raise ValueError('canonical must be a phone list; [] explicitly means unknown')
        if word.get('duration') is not None and word['duration'] < 0:
            raise ValueError('Negative word duration')
    if not isinstance(example.get('observed_phones'), list):
        raise ValueError('observed_phones must be a phone-symbol list in the SAME inventory as canonical')
    if training:
        if len(example.get('word_labels', [])) != len(words) or len(example.get('slots', [])) != len(words) + 1:
            raise ValueError('Targets must have n word BIO labels and n+1 insertion slots')
        if any(label not in BIO for label in example['word_labels']):
            raise ValueError('Unknown BIO label')
        for slot in example['slots']:
            for unit in slot:
                if unit.get('label') not in LABELS or not unit.get('text'):
                    raise ValueError('Each slot unit needs a supported label and nonempty text')
    return example


def load_lexicon(path):
    lexicon = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith(('#', 'file:')):
            continue
        word, *phones = line.split()
        lexicon.setdefault(word.lower(), phones)
    return lexicon


def prepare_ms98(processed, lexicon, *, max_utterances=None, max_words=16,
                 keep_probability=.3, deletion=.08, substitution=.14, insertion=.04, seed=42,
                 max_per_conversation=None):
    """Never describe these dictionary-derived phones as recognized audio.

    Drop labelled tokens to construct ASR-like words. The full spoken stream is
    lexicon-derived, with optional independent phone noise. These weak labels
    remain the project's rule labels, not Treebank/NXT gold annotations.
    """
    if any(limit is not None and limit < 1 for limit in [max_utterances, max_per_conversation]):
        raise ValueError('Preparation limits must be positive')
    if max_words < 1 or not 0 <= keep_probability <= 1:
        raise ValueError('Invalid window size or keep probability')
    if min(deletion, substitution, insertion) < 0 or max(deletion, substitution, insertion) > 1:
        raise ValueError('Noise probabilities must lie in [0, 1]')
    inventory = sorted({p for phones in lexicon.values() for p in phones})
    count = 0
    per_conversation = Counter()
    for side in read_jsonl(processed):
        if max_per_conversation is not None and per_conversation[str(side['conv'])] >= max_per_conversation:
            continue
        for utt, group in groupby(side['tokens'], key=lambda t: t['utt']):
            toks = [t for t in group if t['kind'] != 'NOISE']
            # Source windows preserve utterance boundaries, including all-disfluent windows.
            for offset in range(0, len(toks), max_words):
                if max_per_conversation is not None and per_conversation[str(side['conv'])] >= max_per_conversation:
                    break
                window = toks[offset:offset + max_words]
                uid = f'{utt}:{offset}'
                local_seed = int(hashlib.sha256(f'{seed}:{uid}'.encode()).hexdigest()[:16], 16)
                rng = random.Random(local_seed)
                words, labels, slots, pending, spoken = [], [], [], [], []
                previous_label = 'O'
                for t in window:
                    label = t['label']
                    text = (t['spoken'] or t['raw']).lower()
                    spoken.extend(t['phones'] or [])
                    keep = label not in REMOVABLE or rng.random() < keep_probability
                    if not keep:
                        pending.append({'label': label, 'text': text})
                        continue
                    slots.append(pending)
                    pending = []
                    start, end = t.get('start'), t.get('end')
                    # Canonical phones come ONLY from observed text, not a gold pronunciation target.
                    canonical = lexicon.get(text, [])
                    words.append({'text': text, 'canonical': canonical,
                                  'duration': max(0., end-start) if start is not None and end is not None else None,
                                  'pause_before': t.get('pause_before') or 0.})
                    labels.append('O' if label == 'O' else ('I-' if label == previous_label else 'B-') + label)
                    previous_label = label
                if not words:
                    # Retain one observed token so filler-only turns are represented, not dropped.
                    t = window[-1]
                    text = (t['spoken'] or t['raw']).lower()
                    words = [{'text': text, 'canonical': lexicon.get(text, []),
                              'duration': max(0., t['end']-t['start']), 'pause_before': t.get('pause_before') or 0.}]
                    slots = [pending[:-1]]
                    pending = []
                    labels = ['B-' + t['label']]
                slots.append(pending)
                noisy = []
                for phone in spoken:
                    if rng.random() < deletion:
                        continue
                    noisy.append(rng.choice(inventory) if rng.random() < substitution else phone)
                    if rng.random() < insertion:
                        noisy.append(rng.choice(inventory))
                example = {'id': uid, 'conversation': str(side['conv']), 'word_origin': 'simulated',
                           'phone_source': 'simulated_lexicon', 'label_source': 'ms98_project_rules',
                           'words': words, 'observed_phones': noisy, 'word_labels': labels, 'slots': slots}
                validate(example, training=True)
                yield example
                count += 1
                per_conversation[str(side['conv'])] += 1
                if max_utterances is not None and count >= max_utterances:
                    return


def decode(example, predicted):
    """Restore text and token labels. Predictions have the same shape as targets()."""
    n = len(example['words'])
    if len(predicted) != 2*n + 1:
        raise ValueError('Prediction count must be 2*n+1')
    word_tags = [p.removeprefix('word:') for p in predicted[:n]]
    out_words, out_tags, inserted, observed_index = [], [], [], []
    abstained_copies = 0
    for i in range(n+1):
        last_label = None
        for unit in slot_units(predicted[n+i].removeprefix('gap:')):
            if 'copy' in unit:
                pointer = i + unit['copy']
                if not 0 <= pointer < n:
                    abstained_copies += 1
                    continue
                text = example['words'][pointer]['text']
            else:
                text = unit['text']
            parts = text.split()
            for j, text in enumerate(parts):
                prefix = 'I-' if last_label == unit['label'] or j else 'B-'
                out_words.append(text)
                out_tags.append(prefix + unit['label'])
                inserted.append(True)
                observed_index.append(None)
            last_label = unit['label']
        if i < n:
            out_words.append(example['words'][i]['text'])
            out_tags.append(word_tags[i])
            inserted.append(False)
            observed_index.append(i)
    return {'id': example['id'], 'words': out_words, 'tags': out_tags,
            'text': ' '.join(out_words), 'inserted': inserted,
            'observed_index': observed_index, 'abstained_copies': abstained_copies}
