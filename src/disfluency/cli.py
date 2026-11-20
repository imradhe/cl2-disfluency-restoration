"""Transcript-only preparation, training, restoration, and CL2 ablations."""
import argparse
import json
from pathlib import Path

from .data import load_lexicon, prepare_ms98, read_jsonl, write_jsonl, validate, decode
from .experiment import cross_validate, load_model, make_model, save_model


def common(parser):
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--representations', nargs='+', choices=['tfidf', 'ppmi', 'static', 'contextual', 'sentence'], default=['tfidf'])
    parser.add_argument('--vectors', type=Path, help='Word2Vec/fastText text vectors for the static representation')
    parser.add_argument('--encoder', default='bert-base-uncased')
    parser.add_argument('--cache-dir', type=Path, default=Path('.cache/hf/hub'))
    parser.add_argument('--tiny', action='store_true', help='Random tiny BERT for engineering smoke tests, never paper results')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--parser-model', help='Optional installed spaCy dependency parser for condition E')


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Build explicitly simulated transcript restoration examples')
    prepare.add_argument('--processed', type=Path, default=Path('data/processed/swb_tokens.jsonl.gz'))
    prepare.add_argument('--lexicon', type=Path, default=Path('data/raw/switchboard_ms98/sw-ms98-dict.text'))
    prepare.add_argument('--out', type=Path, default=Path('data/restoration/examples.jsonl.gz'))
    prepare.add_argument('--max-utterances', type=int)
    prepare.add_argument('--max-per-conversation', type=int, help='Cap each conversation when preparing a small diverse sample')
    prepare.add_argument('--max-words', type=int, default=8)
    prepare.add_argument('--keep-probability', type=float, default=.3)
    prepare.add_argument('--deletion', type=float, default=.08)
    prepare.add_argument('--substitution', type=float, default=.14)
    prepare.add_argument('--insertion', type=float, default=.04)
    prepare.add_argument('--seed', type=int, default=42)
    prepare.add_argument('--parser-model')
    run = sub.add_parser('run', help='Run shared conversation-grouped cross validation')
    run.add_argument('--config', type=Path)
    run.add_argument('--data', type=Path, default=Path('data/restoration/examples.jsonl.gz'))
    run.add_argument('--out', type=Path, default=Path('results/restoration'))
    run.add_argument('--conditions', nargs='+', choices=list('ABCDEF'), default=['A', 'B', 'C', 'D'])
    run.add_argument('--models', nargs='+', choices=['rule', 'logistic', 'multinomial', 'svm', 'mlp', 'glmm', 'bert'], default=['rule', 'logistic', 'multinomial'])
    run.add_argument('--balancing', nargs='+', choices=['none', 'under', 'smote', 'cluster'], default=['none'])
    run.add_argument('--folds', type=int, default=10)
    run.add_argument('--fold-manifest', type=Path)
    run.add_argument('--f-base', choices=['D', 'E'], default='D')
    run.add_argument('--save-models', action='store_true')
    run.add_argument('--checkpoints', type=Path, default=Path('ckpt/restoration'))
    common(run)
    train = sub.add_parser('train', help='Fit a final model to the supplied training examples')
    train.add_argument('--data', type=Path, required=True)
    train.add_argument('--out', type=Path, required=True)
    train.add_argument('--condition', choices=list('ABCDE'), default='B')
    train.add_argument('--model', choices=['logistic', 'multinomial', 'svm', 'mlp', 'glmm', 'bert'], default='multinomial')
    train.add_argument('--balancing', choices=['none', 'under', 'smote', 'cluster'], default='none')
    common(train)
    predict = sub.add_parser('predict', help='Restore/tag text using a fitted model; no audio stage')
    predict.add_argument('--model-dir', type=Path, required=True)
    input_group = predict.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=Path, help='JSONL observations without gold targets')
    input_group.add_argument('--text', help='Plain transcript; its own phones contain no information about omitted words')
    predict.add_argument('--lexicon', type=Path, default=Path('data/raw/switchboard_ms98/sw-ms98-dict.text'))
    predict.add_argument('--out', type=Path, required=True)
    predict.add_argument('--device', default='cpu')
    predict.add_argument('--threads', type=int, default=2)
    predict.add_argument('--parser-model')
    return parser


def parsed(examples, parser_model):
    if not parser_model:
        return examples
    import spacy
    from .features import add_dependencies
    nlp = spacy.load(parser_model)
    return [add_dependencies(ex, nlp) for ex in examples]


def main(argv=None):
    parser = get_parser()
    # YAML provides defaults, explicit CLI flags still win.
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.command == 'run' and preliminary.config:
        import yaml
        config = yaml.safe_load(preliminary.config.read_text())
        allowed = {a.dest for a in parser._subparsers._group_actions[0].choices['run']._actions}
        unknown = set(config)-allowed
        if unknown:
            parser.error(f'Unknown config keys: {sorted(unknown)}')
        parser._subparsers._group_actions[0].choices['run'].set_defaults(**config)
    args = parser.parse_args(argv)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=getattr(args, 'threads', 2))
    if getattr(args, 'threads', 2) < 1:
        parser.error('--threads must be positive')
    if (args.command == 'predict' and json.loads((args.model_dir/'kind.json').read_text())['kind'] == 'bert') or getattr(args, 'model', None) == 'bert' or 'bert' in getattr(args, 'models', []) or any(r in ['contextual', 'sentence'] for r in getattr(args, 'representations', [])):
        import torch
        torch.set_num_threads(getattr(args, 'threads', 2))
    if args.command == 'prepare':
        examples = list(prepare_ms98(args.processed, load_lexicon(args.lexicon),
                                    max_utterances=args.max_utterances, max_words=args.max_words,
                                    max_per_conversation=args.max_per_conversation,
                                    keep_probability=args.keep_probability, deletion=args.deletion,
                                    substitution=args.substitution, insertion=args.insertion, seed=args.seed))
        examples = parsed(examples, args.parser_model)
        write_jsonl(args.out, examples)
        meta = {'examples': len(examples), 'conversations': len({e['conversation'] for e in examples}),
                'phone_source': 'simulated_lexicon', 'label_source': 'ms98_project_rules',
                'audio_used': False, 'parser_model': args.parser_model, 'noise': {'deletion': args.deletion, 'substitution': args.substitution, 'insertion': args.insertion}}
        Path(str(args.out)+'.meta.json').write_text(json.dumps(meta, indent=2)+'\n')
        print(json.dumps(meta, indent=2))
        return
    if args.command in ['run', 'train']:
        examples = parsed(list(read_jsonl(args.data)), args.parser_model)
        for ex in examples:
            validate(ex, training=True)
        if args.command == 'run':
            report = cross_validate(examples, args)
            print(json.dumps(report['summary'], indent=2))
        else:
            if len(args.representations) != 1:
                parser.error('train takes exactly one --representations value')
            model = make_model(args.model, args.condition, args.balancing, args, args.representations[0]).fit(examples)
            save_model(model, args.out)
            print(f'Saved {args.model} model to {args.out}')
        return
    model = load_model(args.model_dir, args.device)
    if args.text is not None:
        lexicon = {} if model.builder.condition == 'A' else load_lexicon(args.lexicon)
        words = [{'text': text.lower(), 'canonical': lexicon.get(text.lower(), [])} for text in args.text.split()]
        examples = [{'id': 'text', 'conversation': 'inference', 'words': words,
                     'observed_phones': [p for w in words for p in w['canonical']],
                     'phone_source': 'same_text_lexicon', 'word_origin': 'user_transcript'}]
    else:
        examples = list(read_jsonl(args.input))
    examples = parsed(examples, args.parser_model)
    for ex in examples:
        validate(ex)
    predictions = model.predict(examples)
    restored = [decode(ex, pred) for ex, pred in zip(examples, predictions)]
    for ex, result in zip(examples, restored):
        result['phone_source'] = ex.get('phone_source', 'unspecified')
        if ex.get('phone_source') == 'same_text_lexicon':
            result['limitation'] = 'Text-derived phones provide no independent evidence for omitted words; insertions are statistical predictions.'
    write_jsonl(args.out, restored)
    print('\n'.join(r['text'] for r in restored))
