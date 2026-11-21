"""Frozen versus task-trained embeddings on the full transcript classification data.

Uses exactly the conversation folds and candidate contexts of classification.py.
No durations, phones, audio, or target words are provided to these models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn import functional as F

from .classification import (balanced_test_indices, export_tables, file_hash,
                             metrics, prepare, task_labels)


def digest_parameters(parameters):
    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FoldVocabulary:
    """Map global cache identifiers to training-only words; unseen words become UNK."""
    def fit(self, contexts, train, vocabulary, min_frequency=1):
        counts = np.zeros(len(vocabulary), dtype=np.int64)
        for start in range(0, len(train), 65536):
            counts += np.bincount(np.asarray(contexts[train[start:start+65536]]).ravel(),
                                  minlength=len(vocabulary))
        keep = np.flatnonzero(counts >= min_frequency)
        keep = np.unique(np.r_[0, 1, keep])
        self.words = ['<unk>'] + [vocabulary[i] for i in keep]
        self.mapping = np.zeros(len(vocabulary), dtype=np.int64)
        self.mapping[keep] = np.arange(1, len(keep)+1)
        return self

    def transform(self, contexts):
        return self.mapping[contexts]


def initial_vectors(words, dimension, seed, path=None):
    """Stream a headered .vec or headerless GloVe text file, retaining fold words only."""
    known, imported = {}, 0
    if path:
        wanted = set(words)
        file_dimension = None
        with open(path, encoding='utf8') as stream:
            for line_number, line in enumerate(stream, 1):
                fields = line.split()
                if not fields:
                    continue
                if line_number == 1 and len(fields) == 2 and all(v.isdigit() for v in fields):
                    file_dimension = int(fields[1])
                    continue
                word, *values = fields
                if file_dimension is None:
                    file_dimension = len(values)
                if word.lower() not in wanted:
                    continue
                vector = np.asarray(values, dtype=np.float32)
                if not file_dimension or len(vector) != file_dimension or not np.isfinite(vector).all():
                    raise ValueError(f'Invalid vector at {path}:{line_number}')
                known[word.lower()] = vector
                imported += 1
        if not imported:
            raise ValueError('No pretrained vectors match the training vocabulary')
        dimension = file_dimension
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, .05, (len(words), dimension)).astype(np.float32)
    for i, word in enumerate(words):
        if word in known:
            weights[i] = known[word]
    return weights, {'dimension': dimension, 'matched_words': len(known),
                     'vocabulary_words': len(words),
                     'initialization': 'pretrained_text_vectors' if path else 'random'}


def classification_head(size, classes, hidden, dropout, kind):
    if kind == 'mlp':
        return nn.Sequential(nn.Linear(size,hidden),nn.ReLU(),nn.Dropout(dropout),
                             nn.Linear(hidden,classes))
    return nn.Sequential(nn.Linear(size,classes))


def classification_losses(logits, y, kind):
    if kind in ['multinomial','mlp']:
        return F.cross_entropy(logits,y,reduction='none')
    signs = F.one_hot(y,num_classes=logits.shape[1]).to(logits.dtype)
    if kind == 'logistic':
        return F.binary_cross_entropy_with_logits(logits,signs,reduction='none').mean(-1)
    if kind == 'svm':
        return (1-(2*signs-1)*logits).clamp_min(0).mean(-1)
    raise ValueError(f'Unknown classifier: {kind}')


class WordContextClassifier(nn.Module):
    def __init__(self, weights, classes, hidden=128, frozen=False, dropout=.1, classifier='mlp'):
        super().__init__()
        self.encoder = nn.Embedding.from_pretrained(torch.as_tensor(weights), freeze=frozen)
        self.classifier_kind = classifier
        self.head = classification_head(4*weights.shape[1],classes,hidden,dropout,classifier)

    def forward(self, ids):
        return self.head(self.encoder(ids).flatten(1))

    def embedding_parameters(self):
        return self.encoder.parameters()


def make_word_model(args, initial, classes, frozen, contexts, train, fold_vocab):
    if args.classifier=='glmm':
        from .mixed import VariationalWordGLMM, group_mapping
        mapping = group_mapping(contexts,train,fold_vocab,args.random_min_count)
        return VariationalWordGLMM(initial.copy(),classes,mapping)
    return WordContextClassifier(initial.copy(),classes,args.hidden,frozen,classifier=args.classifier)


class TextContextClassifier(nn.Module):
    """Differentiate through the complete transformer/SentenceTransformer encoder."""
    def __init__(self, model, classes, kind='transformer', hidden=128, frozen=False,
                 cache_dir='.cache/hf/hub', local_only=True, max_length=128, dropout=.1, classifier='mlp'):
        super().__init__()
        self.kind, self.frozen, self.max_length = kind, frozen, max_length
        if kind == 'sentence':
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(str(model), cache_folder=str(cache_dir),
                                               local_files_only=local_only, device='cpu')
            dimension_method = getattr(self.encoder, 'get_embedding_dimension', None)
            size = dimension_method() if dimension_method else self.encoder.get_sentence_embedding_dimension()
            self.tokenizer = self.encoder.tokenizer
            self.max_length = min(max_length, self.encoder.max_seq_length)
        else:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(model), cache_dir=str(cache_dir),
                                                          local_files_only=local_only)
            self.encoder = AutoModel.from_pretrained(str(model), cache_dir=str(cache_dir),
                                                     local_files_only=local_only)
            size = self.encoder.config.hidden_size
            self.max_length = min(max_length, self.encoder.config.max_position_embeddings)
        self.encoder.requires_grad_(not frozen)
        self.classifier_kind = classifier
        self.head = classification_head(size,classes,hidden,dropout,classifier)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def encode_contexts(self, contexts, vocabulary, device):
        # A mask marks the withheld candidate, without exposing its word or label.
        marker = self.tokenizer.mask_token or '|'
        texts = [' '.join([vocabulary[a], vocabulary[b], marker,
                           vocabulary[c], vocabulary[d]]) for a,b,c,d in contexts]
        # Four corpus words can split into many subwords. Fail instead of silently
        # dropping context; shorten source words explicitly if a new corpus needs it.
        encoded = self.tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
        if encoded['input_ids'].shape[1] > self.max_length:
            raise ValueError(f'Context exceeds --max-length={self.max_length}; no silent truncation')
        return {key: value.to(device) for key,value in encoded.items()}

    def forward(self, encoded):
        if self.kind == 'sentence':
            # encode() is inference-only. forward() preserves encoder gradients and
            # the checkpoint's actual pooling/dense/normalization modules.
            pooled = self.encoder(dict(encoded))['sentence_embedding']
        else:
            result = self.encoder(**encoded).last_hidden_state
            mask = encoded['attention_mask'].unsqueeze(-1)
            pooled = (result*mask).sum(1)/mask.sum(1).clamp(min=1)
        return self.head(pooled)

    def embedding_parameters(self):
        encoder = self.encoder[0].auto_model if self.kind == 'sentence' else self.encoder
        return encoder.get_input_embeddings().parameters()


def make_batch(model, contexts, indices, vocabulary, fold_vocab, device):
    values = np.asarray(contexts[indices])
    if isinstance(model, WordContextClassifier):
        return torch.as_tensor(fold_vocab.transform(values), dtype=torch.long, device=device)
    return model.encode_contexts(values, vocabulary, device)


def atomic_torch_save(value,path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def fit_model(model, contexts, labels, train, vocabulary, fold_vocab, *, epochs=3,
              batch_size=1024, encoder_lr=.001, head_lr=.001, balancing='balanced',
              seed=42, device='cpu', state_path=None, resume_state=False,
              checkpoint_every=500, progress_every=100, progress_path=None, amp=False,
              portable_resume=False):
    """Every training row participates in every epoch; weights are train-fold-only."""
    model.to(device)
    counts = np.bincount(labels[train], minlength=model.head[-1].out_features)
    class_weights = (len(train)/(len(counts)*np.maximum(counts,1)) if balancing == 'balanced'
                     else np.ones(len(counts)))
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    parameter_groups = [{'params':encoder_parameters,'lr':encoder_lr},
                        {'params':model.head.parameters(),'lr':head_lr}]
    if hasattr(model,'additional_parameters'):
        parameter_groups.append({'params':model.additional_parameters(),'lr':head_lr,'weight_decay':0.})
    optimizer = torch.optim.AdamW(parameter_groups,weight_decay=.01)
    device_kind = torch.device(device).type
    if amp and device_kind not in ['cuda','mps']:
        raise ValueError('Mixed precision requires a CUDA or MPS device')
    scaler = torch.amp.GradScaler(device_kind,enabled=amp)
    before = digest_parameters(model.encoder.parameters())
    head_before = digest_parameters(model.head.parameters())
    embedding_before = digest_parameters(model.embedding_parameters())
    rng, history = np.random.default_rng(seed), []
    start_epoch,offset,global_step = 0,0,0
    saved_order,loss_sum,examples_seen = None,0.,0
    previous_epoch_seconds = 0.
    signature = {'train_rows_sha256':hashlib.sha256(np.asarray(train,dtype=np.int64).tobytes()).hexdigest(),
                 'train_class_counts':counts.tolist(),'epochs':epochs,'batch_size':batch_size,
                 'encoder_lr':encoder_lr,'head_lr':head_lr,'balancing':balancing,'seed':seed,
                 'amp':amp,'classifier':model.classifier_kind,'encoder_initial':before,'head_initial':head_before}
    if resume_state and state_path and Path(state_path).exists():
        state = torch.load(state_path,map_location='cpu',weights_only=True)
        if portable_resume:
            # A complete checkpoint replaces the freshly constructed network.
            # CPU initializer sequences can differ across PyTorch releases; their
            # hashes do not affect continuation from the actual saved weights.
            for key in ['encoder_initial','head_initial']:
                signature[key] = state['signature'][key]
            before,head_before = signature['encoder_initial'],signature['head_initial']
        if state['signature']!=signature:
            raise ValueError('Training checkpoint does not match the dataset/initialization/training recipe')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
        if device_kind=='mps' and state.get('mps_rng') is not None:
            torch.mps.set_rng_state(state['mps_rng'])
        if device_kind=='cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_epoch,offset,global_step = state['epoch'],state['offset'],state['global_step']
        saved_order = state['order'].numpy() if state['order'] is not None else None
        loss_sum,examples_seen = state['loss_sum'],state['examples_seen']
        previous_epoch_seconds,history = state['epoch_seconds'],state['history']
        print(f'  resumed epoch {start_epoch+1}, offset {offset:,}, step {global_step}',flush=True)

    def checkpoint(epoch,next_offset,order,epoch_seconds):
        if not state_path:
            return
        atomic_torch_save({'signature':signature,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                           'scaler':scaler.state_dict(),'numpy_rng':rng.bit_generator.state,
                           'torch_rng':torch.get_rng_state(),
                           'mps_rng':torch.mps.get_rng_state() if device_kind=='mps' else None,
                           'cuda_rng':torch.cuda.get_rng_state_all() if device_kind=='cuda' else None,
                           'epoch':epoch,'offset':next_offset,'global_step':global_step,
                           'order':torch.from_numpy(order) if order is not None else None,
                           'loss_sum':loss_sum,'examples_seen':examples_seen,'epoch_seconds':epoch_seconds,
                           'history':history},state_path)

    for epoch in range(start_epoch,epochs):
        model.train()
        if epoch==start_epoch and saved_order is not None:
            order = saved_order
        else:
            order,loss_sum,examples_seen = rng.permutation(train),0.,0
            offset,previous_epoch_seconds = 0,0.
        began = time.monotonic()
        for start in range(offset,len(order),batch_size):
            selected = order[start:start+batch_size]
            batch = make_batch(model, contexts, selected, vocabulary, fold_vocab, device)
            y = torch.as_tensor(np.asarray(labels[selected], dtype=np.int64), device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_kind,dtype=torch.float16) if amp else nullcontext():
                losses = (model.training_losses(batch,y) if hasattr(model,'training_losses')
                          else classification_losses(model(batch),y,model.classifier_kind))
                # Global inverse-frequency weights have mean one on training rows.
                loss = (losses*weights[y]).mean()
                if hasattr(model,'regularization'):
                    loss = loss+model.regularization(len(train))
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())*len(selected)
            examples_seen += len(selected)
            global_step += 1
            elapsed = previous_epoch_seconds+time.monotonic()-began
            if global_step%progress_every==0:
                progress = {'phase':'training','epoch':epoch+1,'epochs':epochs,'global_step':global_step,
                            'examples_seen_this_epoch':examples_seen,'training_rows':len(train),
                            'epoch_seconds':elapsed,'loss_so_far':loss_sum/examples_seen,
                            'rows_per_second':examples_seen/max(elapsed,1e-6),
                            'training_state':str(state_path) if state_path else None}
                if progress_path:
                    Path(progress_path).write_text(json.dumps(progress,indent=2)+'\n')
                print(f"  step {global_step}: epoch {epoch+1}, {examples_seen:,}/{len(train):,} rows, "
                      f"{progress['rows_per_second']:.1f} rows/s",flush=True)
            if state_path and (global_step==1 or global_step%checkpoint_every==0):
                checkpoint(epoch,start+len(selected),order,elapsed)
        history.append({'epoch': epoch+1, 'loss': loss_sum/examples_seen,
                        'examples_seen': examples_seen,'seconds':previous_epoch_seconds+time.monotonic()-began})
        print(f"  epoch {epoch+1}: {examples_seen:,} rows, loss {history[-1]['loss']:.5f}, "
              f"{history[-1]['seconds']:.1f}s", flush=True)
        loss_sum,examples_seen = 0.,0
        checkpoint(epoch+1,0,None,0.)
        offset,saved_order = 0,None
    after = digest_parameters(model.encoder.parameters())
    if not encoder_parameters and before != after:
        raise AssertionError('Frozen encoder changed')
    if encoder_parameters and before == after:
        raise AssertionError('Trainable encoder did not update')
    report = {'history': history, 'train_class_counts': counts.tolist(),
            'encoder_trainable_parameters': sum(p.numel() for p in encoder_parameters),
            'encoder_sha256_before': before, 'encoder_sha256_after': after,
            'head_sha256_before': head_before,
            'encoder_changed': before != after,
            'embedding_weights_changed': embedding_before != digest_parameters(model.embedding_parameters())}
    if hasattr(model,'diagnostics'):
        report['diagnostics'] = model.diagnostics()
    return report


@torch.no_grad()
def predict_model(model, contexts, test, vocabulary, fold_vocab, batch_size=1024, device='cpu'):
    model.eval()
    predicted = np.empty(len(test), dtype=np.int64)
    for start in range(0, len(test), batch_size):
        selected = test[start:start+batch_size]
        batch = make_batch(model, contexts, selected, vocabulary, fold_vocab, device)
        predicted[start:start+len(selected)] = model(batch).argmax(-1).cpu().numpy()
    return predicted


def save_checkpoint(model, path, fold_vocab, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if isinstance(model, TextContextClassifier):
        model.encoder.save_pretrained(path/'encoder')
        model.tokenizer.save_pretrained(path/'encoder')
    else:
        np.save(path/'mapping.npy', fold_vocab.mapping)
        (path/'vocabulary.json').write_text(json.dumps(fold_vocab.words)+'\n')
    torch.save(model.state_dict() if isinstance(model, WordContextClassifier) else model.head.state_dict(),
               path/'weights.pt')
    (path/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')


def load_checkpoint(path, device='cpu'):
    path = Path(path)
    metadata = json.loads((path/'metadata.json').read_text())
    options = metadata['model']
    fold_vocab = None
    if options['kind'] in ['random','static']:
        fold_vocab = FoldVocabulary()
        fold_vocab.words = json.loads((path/'vocabulary.json').read_text())
        fold_vocab.mapping = np.load(path/'mapping.npy')
        state = torch.load(path/'weights.pt',map_location='cpu',weights_only=True)
        initial = np.zeros((len(fold_vocab.words),options['dimension']),np.float32)
        if options.get('classifier')=='glmm':
            from .mixed import VariationalWordGLMM
            model = VariationalWordGLMM(initial,options['classes'],state['group_mapping'].numpy(),len(state['nodes']))
        else:
            model = WordContextClassifier(initial,options['classes'],options['hidden'],options['frozen'],
                                          classifier=options.get('classifier','mlp'))
        model.load_state_dict(state)
    else:
        model = TextContextClassifier(path/'encoder', options['classes'], kind=options['kind'],
                                      hidden=options['hidden'], frozen=options['frozen'],
                                      max_length=options['max_length'],classifier=options.get('classifier','mlp'))
        model.head.load_state_dict(torch.load(path/'weights.pt', map_location='cpu', weights_only=True))
    return model.to(device).eval(), fold_vocab, metadata


def conversation_folds(groups, labels, count, path=None):
    mapping = {}
    for fold,(_,test) in enumerate(GroupKFold(n_splits=count).split(np.zeros(len(groups)),labels,groups)):
        mapping.update({str(int(group)): fold for group in np.unique(groups[test])})
    if path:
        source = json.loads(Path(path).read_text())
        if source['folds'] != count or source['groups'] != mapping:
            raise ValueError('Requested folds differ from baseline conversation folds')
    return np.asarray([mapping[str(int(g))] for g in groups], dtype=np.uint16),mapping


def aggregate(results, manifest, settings, directory):
    summary = []
    for view,task in sorted({(r['view'],r['task']) for r in results}):
        rows = [r for r in results if r['view'] == view and r['task'] == task]
        part = {'view': view, 'task': task, 'folds': len(rows),
                'test_examples_total': sum(r['natural_prevalence']['examples'] for r in rows)}
        for evaluation in ['natural_prevalence','balanced_test']:
            if evaluation not in rows[0]:
                continue
            def std(values):
                return float(np.std(values,ddof=1)) if len(values)>1 else 0.
            score = {}
            for field in ['accuracy','macro_f1']:
                values = [r[evaluation][field] for r in rows]
                score[field] = {'mean': float(np.mean(values)), 'std': std(values)}
            score['per_class'] = {}
            for name in rows[0][evaluation]['per_class']:
                stats = {'support': sum(r[evaluation]['per_class'][name]['support'] for r in rows)}
                for field in ['precision','recall','f1']:
                    values = [r[evaluation]['per_class'][name][field] for r in rows]
                    stats.update({field+'_mean': float(np.mean(values)), field+'_std': std(values)})
                score['per_class'][name] = stats
            score['pooled_confusion'] = np.sum([r[evaluation]['confusion'] for r in rows],axis=0).tolist()
            part[evaluation] = score
        summary.append(part)
    complete = all({r['fold'] for r in results if r['view']==view and r['task']==task} == set(range(settings['folds']))
                   for view in settings['views'] for task in settings['tasks'])
    report = {'data': manifest, 'settings': settings, 'complete': complete, 'summary': summary,
              'notes': ['Transcripts only; no timing, phones, audio or target words.',
                        'Contexts and known candidate positions match the full-corpus classification baseline.',
                        'Random embeddings are trained from scratch; only pretrained initializations are fine-tuning.',
                        'No test-fold fitting, resampling, early stopping or hyperparameter selection.',
                        'Frozen and updated models start from identical encoder and head weights per fold/task.']}
    (directory/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    export_tables(report,directory)
    return report


def fit_final_models(args, arrays, manifest, vocabulary):
    """Refit the fixed recipe on all rows for use; never report this as held-out scoring."""
    contexts,labels = arrays['contexts'],arrays['labels']
    artifacts = []
    for task in args.tasks:
        mask,y,names = task_labels(labels,task)
        train = np.flatnonzero(mask)
        fold_vocab = None
        initialization = {'initialization':'pretrained_'+args.embedding}
        if args.embedding in ['random','static']:
            fold_vocab = FoldVocabulary().fit(contexts,train,vocabulary,args.min_frequency)
            initial,initialization = initial_vectors(fold_vocab.words,args.dimension,args.seed,args.vectors)
        for mode in args.modes:
            torch.manual_seed(args.seed)
            frozen = mode == 'frozen'
            if args.embedding in ['random','static']:
                model = make_word_model(args,initial,len(names),frozen,contexts,train,fold_vocab)
                encoder_lr = args.encoder_lr or .001
            else:
                model = TextContextClassifier(args.encoder,len(names),args.embedding,args.hidden,frozen,
                                              args.hf_cache,not args.allow_download,args.max_length,classifier=args.classifier)
                encoder_lr = args.encoder_lr or .00002
            print(f'Final refit / {task} / {mode}: all {len(train):,} eligible rows',flush=True)
            audit = fit_model(model,contexts,y,train,vocabulary,fold_vocab,epochs=args.epochs,
                              batch_size=args.batch_size,encoder_lr=encoder_lr,head_lr=args.head_lr,
                              balancing=args.balancing,seed=args.seed,device=args.device)
            options = {'kind':args.embedding,'classes':len(names),'hidden':args.hidden,'frozen':frozen,
                       'max_length':args.max_length,'dimension':initialization.get('dimension'),
                       'classifier':args.classifier}
            path = Path(args.checkpoints)/f'full_{args.embedding}_{mode}_{task}'
            metadata = {'model':options,'task':task,'classes':names,'fit':'all eligible corpus rows',
                        'train_examples':len(train),'source_sha256':manifest['source_sha256'],
                        'initialization':initialization,'training':audit,
                        'global_vocabulary':str(Path(args.cache)/'vocabulary.json'),
                        'evaluation':'none; use the separate out-of-fold report'}
            save_checkpoint(model,path,fold_vocab,metadata)
            artifacts.append({'path':str(path),**metadata})
            del model
    (Path(args.out)/'final_models.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    return artifacts


def run(args):
    torch.set_num_threads(args.threads)
    directory = Path(args.out)
    directory.mkdir(parents=True,exist_ok=True)
    arrays,manifest = prepare(args.processed,args.cache)
    vocabulary = json.loads((Path(args.cache)/'vocabulary.json').read_text())
    labels,groups,contexts = arrays['labels'],arrays['groups'],arrays['contexts']
    fold_of_row,mapping = conversation_folds(groups,labels,args.folds,args.fold_manifest)
    views = [args.embedding+'_'+mode for mode in args.modes]
    settings = {key:value for key,value in vars(args).items()
                if key not in ['resume','fold_ids','out','checkpoints','save_checkpoints','fit_final',
                               'training_states','checkpoint_every','progress_every','save_last_fold','portable_resume']}
    settings.update(groups=mapping,source_sha256=manifest['source_sha256'],views=views,
                    vectors_sha256=file_hash(args.vectors) if args.vectors else None,
                    transcript_only=True, sample_limit=None)
    settings_file = directory/'settings.json'
    if settings_file.exists():
        previous = json.loads(settings_file.read_text())
        previous.setdefault('classifier','mlp')
        previous.setdefault('random_min_count',30)
        previous.setdefault('amp',False)
        if previous != settings:
            raise ValueError('Output settings differ; select a new --out directory')
    settings_file.write_text(json.dumps(settings,indent=2)+'\n')
    (directory/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results = []
    selected_folds = args.fold_ids if args.fold_ids is not None else list(range(args.folds))
    for fold in selected_folds:
        for task in args.tasks:
            if args.resume and all((directory/f'{view}_{task}_fold{fold:02d}.json').exists() for view in views):
                continue
            mask,y,names = task_labels(labels,task)
            train = np.flatnonzero((fold_of_row != fold)&mask)
            test = np.flatnonzero((fold_of_row == fold)&mask)
            assert not np.intersect1d(np.unique(groups[train]),np.unique(groups[test])).size
            fold_vocab = None
            initialization = {'initialization': 'pretrained_'+args.embedding}
            if args.embedding in ['random','static']:
                fold_vocab = FoldVocabulary().fit(contexts,train,vocabulary,args.min_frequency)
                initial,initialization = initial_vectors(fold_vocab.words,args.dimension,args.seed+fold,args.vectors)
            for mode,view in zip(args.modes,views):
                path = directory/f'{view}_{task}_fold{fold:02d}.json'
                if args.resume and path.exists():
                    continue
                torch.manual_seed(args.seed+fold)
                frozen = mode == 'frozen'
                if args.embedding in ['random','static']:
                    # Clone so a preceding updated run cannot modify the next initialization.
                    model = make_word_model(args,initial,len(names),frozen,contexts,train,fold_vocab)
                    encoder_lr = args.encoder_lr or .001
                else:
                    model = TextContextClassifier(args.encoder,len(names),args.embedding,args.hidden,frozen,
                                                  args.hf_cache,not args.allow_download,args.max_length,classifier=args.classifier)
                    encoder_lr = args.encoder_lr or .00002
                print(f'{view} / {task} / fold {fold+1}/{args.folds}: train {len(train):,}, test {len(test):,}',flush=True)
                began = time.monotonic()
                audit = fit_model(model,contexts,y,train,vocabulary,fold_vocab,epochs=args.epochs,
                                  batch_size=args.batch_size,encoder_lr=encoder_lr,head_lr=args.head_lr,
                                  balancing=args.balancing,seed=args.seed+fold,device=args.device,
                                  state_path=Path(args.training_states)/path.stem/'state.pt' if args.training_states else None,
                                  resume_state=args.resume,checkpoint_every=args.checkpoint_every,
                                  progress_every=args.progress_every,progress_path=directory/'progress.json',amp=args.amp,
                                  portable_resume=args.portable_resume)
                (directory/'progress.json').write_text(json.dumps({'phase':'evaluating','fold':fold,'task':task,
                                                                  'view':view,'test_examples':len(test)},indent=2)+'\n')
                predicted = predict_model(model,contexts,test,vocabulary,fold_vocab,args.batch_size,args.device)
                record = {'view':view,'task':task,'fold':fold,'classifier':args.classifier,'train_examples':len(train),
                          'natural_prevalence':metrics(y[test],predicted,names),'training':audit,
                          'initialization':initialization,'seconds':time.monotonic()-began}
                if task != 'multiclass':
                    balanced = balanced_test_indices(y[test],args.seed+fold)
                    record['balanced_test'] = metrics(y[test][balanced],predicted[balanced],names)
                if args.save_checkpoints or (args.save_last_fold and fold==args.folds-1):
                    model_options = {'kind':args.embedding,'classes':len(names),'hidden':args.hidden,
                                     'frozen':frozen,'max_length':args.max_length,
                                     'dimension':initialization.get('dimension'),'classifier':args.classifier}
                    save_checkpoint(model,Path(args.checkpoints)/path.stem,fold_vocab,
                                    {'model':model_options,'task':task,'classes':names,'fold':fold,
                                     'source_sha256':manifest['source_sha256']})
                path.write_text(json.dumps(record,indent=2)+'\n')
                if args.training_states:
                    state_file = Path(args.training_states)/path.stem/'state.pt'
                    if state_file.exists():
                        state_file.unlink()  # Completed model/report supersedes optimizer state.
                (directory/'progress.json').write_text(json.dumps({'phase':'fold_complete','fold':fold,
                                                                  'task':task,'view':view,'report':str(path)},indent=2)+'\n')
                print(f"  accuracy {record['natural_prevalence']['accuracy']:.4f}, "
                      f"macro F1 {record['natural_prevalence']['macro_f1']:.4f}, "
                      f"embedding changed={audit['embedding_weights_changed']}",flush=True)
                del model
    for view in views:
        for task in args.tasks:
            for fold in range(args.folds):
                path = directory/f'{view}_{task}_fold{fold:02d}.json'
                if path.exists():
                    results.append(json.loads(path.read_text()))
    report = aggregate(results,manifest,settings,directory)
    (directory/'progress.json').write_text(json.dumps({'phase':'complete' if report['complete'] else 'partial',
                                                      'completed_fold_fits':len(results)},indent=2)+'\n')
    if args.fit_final:
        if not report['complete']:
            raise ValueError('Complete all cross-validation folds before --fit-final')
        fit_final_models(args,arrays,manifest,vocabulary)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed',default='data/processed/swb_tokens.jsonl.gz')
    parser.add_argument('--cache',default='data/classification/full')
    parser.add_argument('--out',default='results/classification/embeddings')
    parser.add_argument('--embedding',choices=['random','static','transformer','sentence'],default='random')
    parser.add_argument('--classifier',choices=['logistic','multinomial','svm','mlp','glmm'],default='mlp')
    parser.add_argument('--random-min-count',type=int,default=30)
    parser.add_argument('--vectors',help='Headered Word2Vec/fastText .vec or headerless GloVe text')
    parser.add_argument('--encoder',default='bert-base-uncased')
    parser.add_argument('--hf-cache',default='.cache/hf/hub')
    parser.add_argument('--allow-download',action='store_true')
    parser.add_argument('--modes',nargs='+',choices=['frozen','updated'],default=['frozen','updated'])
    parser.add_argument('--tasks',nargs='+',choices=['binary','multiclass','filler','reparandum'],
                        default=['binary','multiclass','filler','reparandum'])
    parser.add_argument('--folds',type=int,default=10)
    parser.add_argument('--fold-manifest',help='Baseline settings.json; require exactly the same conversation folds')
    parser.add_argument('--fold-ids',type=int,nargs='+',help='Run selected full-size folds; omitted folds remain explicitly incomplete')
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--dimension',type=int,default=64)
    parser.add_argument('--hidden',type=int,default=128)
    parser.add_argument('--min-frequency',type=int,default=1)
    parser.add_argument('--max-length',type=int,default=128)
    parser.add_argument('--encoder-lr',type=float)
    parser.add_argument('--head-lr',type=float,default=.001)
    parser.add_argument('--balancing',choices=['none','balanced'],default='balanced')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--amp',action='store_true',help='Use float16 autocast with gradient scaling on CUDA/MPS')
    parser.add_argument('--training-states',help='Directory for resumable model/optimizer/RNG checkpoints')
    parser.add_argument('--checkpoint-every',type=int,default=500)
    parser.add_argument('--progress-every',type=int,default=100)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--portable-resume',action='store_true',help='Restore complete checkpoints across PyTorch/device environments; retain saved initialization provenance')
    parser.add_argument('--save-checkpoints',action='store_true')
    parser.add_argument('--save-last-fold',action='store_true',help='Keep final-fold models only, reducing disk use for large encoders')
    parser.add_argument('--fit-final',action='store_true',help='After complete CV, refit and save models on all rows; no training-set evaluation')
    parser.add_argument('--checkpoints',default='ckpt/classification_embeddings')
    args = parser.parse_args(argv)
    if args.embedding == 'static' and not args.vectors:
        parser.error('--embedding static requires --vectors')
    if args.embedding != 'static' and args.vectors:
        parser.error('--vectors applies only to --embedding static')
    if args.classifier=='glmm' and (args.embedding not in ['random','static'] or args.modes!=['frozen']):
        parser.error('GLMM requires fixed word features: --embedding static/random --modes frozen')
    if min(args.checkpoint_every,args.progress_every,args.random_min_count)<1:
        parser.error('Positive checkpoint/progress intervals and random-effect minimum counts are required')
    if min(args.epochs,args.batch_size,args.dimension,args.hidden,args.min_frequency,args.threads) < 1 or args.folds < 2:
        parser.error('Positive training sizes and at least two folds are required')
    if args.fold_ids is not None and (not args.fold_ids or min(args.fold_ids)<0 or max(args.fold_ids)>=args.folds):
        parser.error('--fold-ids must lie in [0, folds)')
    if (args.encoder_lr is not None and args.encoder_lr <= 0) or args.head_lr <= 0:
        parser.error('Learning rates must be positive')
    run(args)


if __name__ == '__main__':
    main()
