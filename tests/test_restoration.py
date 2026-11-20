import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from disfluency.alignment import align, phone_evidence
from disfluency.baselines import ClassicalTagger, RandomInterceptLogit
from disfluency.data import decode, prepare_ms98, read_jsonl, targets, validate, write_jsonl
from disfluency.experiment import folds, load_model, save_model
from disfluency.features import FeatureBuilder, dlt
from disfluency.metrics import score_restorations
from disfluency.representations import RepresentationSpace
from disfluency.sampling import TrainingSpace, resample


def example(i=0, filler=True):
    return {'id': f'ex{i}', 'conversation': str(i//2), 'word_origin': 'simulated',
            'phone_source': 'simulated_lexicon', 'label_source': 'fixture',
            'words': [{'text': 'i', 'canonical': ['ay'], 'duration': .2},
                      {'text': 'go', 'canonical': ['g', 'ow'], 'duration': .3}],
            'observed_phones': ['ay'] + (['ah', 'm'] if filler else []) + ['g', 'ow'],
            'heads': [1, -1], 'word_labels': ['O', 'O'],
            'slots': [[], [{'label': 'FILLER_HESITATION', 'text': 'um'}] if filler else [], []]}


def test_alignment_insertion_and_empty_inputs():
    ev = phone_evidence([['ay'], ['g', 'ow']], ['ay', 'ah', 'm', 'g', 'ow'])
    assert ev['gaps'] == [[], ['ah', 'm'], []]
    assert ev['distance'] == 2
    assert ev['words'][1]['realized'] == ['g', 'ow']
    assert [e.op for e in align([], ['a'])] == ['I']
    assert [e.op for e in align(['a'], [])] == ['D']
    assert phone_evidence([[]], ['ax'])['gaps'] == [[], ['ax']]


def test_dlt_matches_published_example():
    # My dog also likes eating sausage
    ic, sc = dlt([1, 3, 3, -1, 3, 4])
    assert ic == [1, 1, 1, 2, 1, 1]
    assert sc == [2, 2, 3, 1, 1, 1]
    with pytest.raises(ValueError, match='cycle'):
        dlt([1, 0])


def test_no_gold_target_information_in_features():
    original = example()
    changed = copy.deepcopy(original)
    changed['slots'] = [[], [], []]
    changed['word_labels'] = ['B-REPETITION', 'I-REPETITION']
    for condition in 'ABCDE':
        builder = FeatureBuilder(condition).fit([original])
        assert builder.rows(original) == builder.rows(changed)
        assert builder.serialize(original, 1, 'word') == builder.serialize(changed, 1, 'word')


def test_fold_statistics_do_not_learn_test_words_or_durations():
    training = example()
    builder = FeatureBuilder('E').fit([training])
    before = dict(builder.duration_sum)
    test = example(2)
    test['words'][1].update(text='unseenword', duration=999.)
    builder.rows(test)
    assert 'unseenword' not in builder.vocab
    assert builder.duration_sum == before


def test_e_requires_real_dependency_parse_and_timing():
    ex = example()
    del ex['heads']
    with pytest.raises(ValueError, match='dependency heads'):
        FeatureBuilder('E').fit([ex]).rows(ex)
    ex['heads'] = [1, -1]
    del ex['words'][0]['duration']
    with pytest.raises(ValueError, match='durations'):
        FeatureBuilder('E').fit([ex]).rows(ex)


def test_copy_targets_generalize_to_new_words():
    ex = example()
    ex['slots'] = [[], [{'label': 'REPETITION', 'text': 'go'}], []]
    encoded = targets(ex)
    assert '"copy":0' in encoded[3]
    ex['words'][1]['text'] = 'walk'
    assert decode(ex, encoded)['text'] == 'i walk walk'


def test_roundtrip_prepared_targets_and_provenance(tmp_path):
    token = lambda text, label, phones, i: {'utt': 'u', 'raw': text, 'spoken': text,
        'kind': 'WORD', 'label': label, 'phones': phones, 'start': i*.2, 'end': (i+1)*.2, 'pause_before': 0}
    side = {'sid': 's', 'conv': '1', 'tokens': [token('i', 'O', ['ay'], 0),
            token('um', 'FILLER_HESITATION', ['ah', 'm'], 1), token('go', 'O', ['g', 'ow'], 2)]}
    path = tmp_path/'source.jsonl.gz'
    write_jsonl(path, [side])
    records = list(prepare_ms98(path, {'i':['ay'], 'um':['ah','m'], 'go':['g','ow']},
                    keep_probability=0, deletion=0, substitution=0, insertion=0))
    ex = records[0]
    assert validate(ex, training=True)
    assert decode(ex, targets(ex))['text'] == 'i um go'
    assert ex['phone_source'] == 'simulated_lexicon'
    assert ex['label_source'] == 'ms98_project_rules'
    assert ex['observed_phones'] == ['ay', 'ah', 'm', 'g', 'ow']
    # Filler-only turns are kept in the task.
    side['tokens'] = [token('um', 'FILLER_HESITATION', ['ah','m'], 0)]
    write_jsonl(path, [side])
    ex = next(prepare_ms98(path, {'um':['ah','m']}, keep_probability=0))
    assert decode(ex, targets(ex))['text'] == 'um'


def test_grouped_folds_reused_and_reject_different_data(tmp_path):
    examples = [example(i, i%2 == 0) for i in range(24)]
    path = tmp_path/'folds.json'
    manifest = folds(examples, 10, manifest=path)
    assert len(set(manifest['groups'].values())) == 10
    assert folds(examples, 10, manifest=path) == manifest
    for fold in range(10):
        train = {e['conversation'] for e in examples if manifest['groups'][e['conversation']] != fold}
        test = {e['conversation'] for e in examples if manifest['groups'][e['conversation']] == fold}
        assert not train & test
    examples[0]['observed_phones'] = []
    with pytest.raises(ValueError, match='different data'):
        folds(examples, 10, manifest=path)


def test_span_matching_is_one_to_one_and_text_sensitive():
    gold = [{'id': 'x', 'words': ['i','um','go'], 'tags': ['O','B-FILLER_HESITATION','O']}]
    assert score_restorations(gold, gold)['exact']['micro']['f1'] == 1.
    wrong = [{'id': 'x', 'words': ['i','uh','go'], 'tags': ['O','B-FILLER_HESITATION','O']}]
    assert score_restorations(gold, wrong)['exact']['micro']['tp'] == 0
    split_gold = [{'id':'x', 'words':['a','b'], 'tags':['B-REPAIR','I-REPAIR']}]
    split_pred = [{'id':'x', 'words':['a','b'], 'tags':['B-REPAIR','B-REPAIR']}]
    scores = score_restorations(split_gold, split_pred)
    assert scores['relaxed']['micro']['tp'] == 1
    assert scores['exact']['micro']['tp'] == 0


def test_restoration_only_excludes_visible_labels():
    ex = example()
    ex['word_labels'][0] = 'B-FILLER_DM'
    gold = decode(ex, targets(ex))
    scores = score_restorations([gold], [gold])
    assert scores['exact']['micro']['support'] == 2
    assert scores['restoration_only']['exact']['micro']['support'] == 1


@pytest.mark.parametrize('policy', ['none', 'under', 'smote', 'cluster'])
def test_sampling_balances_training_rows_and_reports_singletons(policy):
    x = np.arange(18, dtype=float).reshape(6,3)
    y = np.array(['a']*4 + ['b'] + ['c'])
    sampled, labels, origins, report = resample(x, y, policy)
    assert len(sampled) == len(labels) == len(origins)
    assert set(origins) <= set(range(6))
    _, counts = np.unique(labels, return_counts=True)
    if policy != 'none':
        assert len(set(counts)) == 1
    if policy == 'smote':
        assert report['singleton_replication'] == ['b', 'c']


@pytest.mark.parametrize('representation', ['tfidf', 'ppmi', 'static'])
def test_embedding_spaces_are_fitted_without_test_vocabulary(tmp_path, representation):
    vector_path = tmp_path/'tiny.vec'
    vector_path.write_text('3 2\ni 1 0\ngo 0 1\nunseen 1 1\n')
    builder = FeatureBuilder('A').fit([example()])
    rows = builder.rows(example())
    space = RepresentationSpace(representation, vectors=vector_path).fit(rows)
    transformed = space.transform(rows)
    assert transformed.shape[0] == 5
    assert np.isfinite(transformed.data).all()
    if representation == 'tfidf':
        assert 'unseen' not in space.word.vocabulary_
    if representation == 'ppmi':
        assert 'unseen' not in space.text.vocab


@pytest.mark.parametrize('kind', ['logistic', 'multinomial', 'svm', 'mlp', 'glmm'])
def test_classical_models_restore_and_reload_without_gold_input(tmp_path, kind):
    examples = [example(i, i%2 == 0) for i in range(12)]
    model = ClassicalTagger(FeatureBuilder('E'), model=kind, representation='ppmi').fit(examples)
    blind = copy.deepcopy(examples[:2])
    for ex in blind:
        del ex['slots'], ex['word_labels']
    prediction = model.predict(blind)
    assert len(prediction) == 2
    assert all(len(p) == 5 for p in prediction)
    assert all(p.startswith('word:') for p in prediction[0][:2])
    assert all(p.startswith('gap:') for p in prediction[0][2:])
    save_model(model, tmp_path/'checkpoint')
    assert load_model(tmp_path/'checkpoint').predict(blind) == prediction


def test_glmm_random_intercept_distinguishes_groups():
    x = np.zeros((80, 1))
    y = np.array([1.]*35 + [0.]*5 + [1.]*5 + [0.]*35)
    groups = np.array(['high']*40 + ['low']*40)
    model = RandomInterceptLogit().fit(x, y, groups)
    p = model.predict_proba(np.zeros((3,1)), ['high','low','unseen'])
    assert p[0] > p[2] > p[1]
    assert np.isfinite(p).all()


def test_model_learns_to_insert_a_missing_filler():
    training = [example(i, True) for i in range(20)]
    model = ClassicalTagger(FeatureBuilder('A'), model='multinomial').fit(training)
    blind = copy.deepcopy(training[0])
    del blind['word_labels'], blind['slots']
    restored = decode(blind, model.predict([blind])[0])
    assert restored['text'] == 'i um go'
    assert restored['inserted'] == [False, True, False]


def test_tiny_transformer_numeric_features_balancing_and_checkpoint(tmp_path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from disfluency.bert import BertTagger
    torch.set_num_threads(2)
    vocabulary = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'i', 'go', 'um', 'ah', 'm',
                  'ay', 'g', 'ow', '|', '/']
    backend = Tokenizer(WordPiece({t:i for i,t in enumerate(vocabulary)}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]',
                                       cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]')
    encoder_path = tmp_path/'tokenizer'
    tokenizer.save_pretrained(encoder_path)
    training = [example(i, i%2 == 0) for i in range(6)]
    model = BertTagger(FeatureBuilder('E'), encoder=str(encoder_path), tiny=True, epochs=1,
                       batch_size=3, learning_rate=.001, balancing='smote').fit(training)
    blind = copy.deepcopy(training[:2])
    for ex in blind:
        del ex['word_labels'], ex['slots']
    predicted = model.predict(blind)
    assert len(predicted[0]) == 5
    assert model.sampling_report['encoder_initialization'] == 'random_tiny'
    assert model.sampling_report['stage'] == 'train-fold frozen-encoder entity representations'
    save_model(model, tmp_path/'model')
    assert load_model(tmp_path/'model').predict(blind) == predicted


def test_full_classifier_uses_every_lexical_row_and_hides_target(tmp_path):
    from disfluency.classification import prepare, context_matrix, task_labels, metrics
    def token(word,label,i):
        return {'utt':'u','spoken':word,'raw':word,'kind':'WORD','label':label,
                'start':i*.2,'end':(i+1)*.2}
    path=tmp_path/'source.jsonl.gz'
    source=[{'conv':12,'tokens':[token('i','O',0),token('um','FILLER_HESITATION',1),
                                token('go','O',2),token('go','REPETITION',3),token('home','O',4)]}]
    write_jsonl(path,source)
    arrays,manifest=prepare(path,tmp_path/'cache')
    assert manifest['examples']==5
    assert manifest['sample_limit'] is None
    assert manifest['conversations']==1
    assert arrays['labels'].shape==(5,)
    vocab=json.loads((tmp_path/'cache/vocabulary.json').read_text())
    # The missing um is represented only by i/go context, not its own text.
    assert 'um' not in [vocab[i] for i in arrays['contexts'][1]]
    assert context_matrix(arrays['contexts']).shape[0]==5
    mask,y,names=task_labels(arrays['labels'],'binary')
    assert mask.all() and y.tolist()==[0,1,0,1,0]
    filler_mask,filler_y,_=task_labels(arrays['labels'],'filler')
    assert filler_mask.tolist()==[True,True,True,False,True]
    reparandum_mask,reparandum_y,_=task_labels(arrays['labels'],'reparandum')
    assert reparandum_mask.tolist()==[True,False,True,True,True]
    result=metrics(y,y,names)
    assert result['accuracy']==1 and result['macro_f1']==1
    assert sum(p['support'] for p in result['per_class'].values())==5


def test_full_classifier_balanced_eval_does_not_modify_training():
    from disfluency.classification import balanced_test_indices
    y=np.array([0]*100+[1]*10)
    chosen=balanced_test_indices(y,42)
    assert np.bincount(y[chosen]).tolist()==[10,10]
    assert y.tolist()==[0]*100+[1]*10
    assert np.array_equal(chosen,balanced_test_indices(y,42))


def test_full_classification_runner_exports_all_out_of_fold_examples(tmp_path):
    from disfluency.classification import run
    source=[]
    for conversation in range(4):
        source.append({'conv':conversation,'tokens':[
            {'utt':f'u{conversation}','spoken':word,'raw':word,'kind':'WORD','label':label,
             'start':i*.2,'end':(i+1)*.2}
            for i,(word,label) in enumerate([('i','O'),('um','FILLER_HESITATION'),('go','O')])]})
    path=tmp_path/'processed.jsonl.gz'
    write_jsonl(path,source)
    args=SimpleNamespace(processed=path,cache=tmp_path/'cache',out=tmp_path/'out',threads=1,
                         folds=2,seed=42,epochs=2,alpha=1e-6,balancing='balanced',
                         views=['context','context_timing'],tasks=['binary','multiclass'],resume=False)
    report=run(args)
    assert report['data']['examples']==12
    for summary in report['summary']:
        assert summary['test_examples_total']==12
        assert sum(x['support'] for x in summary['natural_prevalence']['per_class'].values())==12
    assert (args.out/'per_class.csv').is_file()
    assert (args.out/'comparison.csv').is_file()
