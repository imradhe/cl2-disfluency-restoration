"""Resume the full BERT experiment on a Colab CUDA runtime with Drive backups.

Run from the prepared notebook after mounting Drive and extracting project.tar.gz.
The training data remain on the VM; reports and rolling checkpoint backups persist
in the user's dedicated Drive folder. No account credentials belong in this file.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def assemble_checkpoint(folder, destination):
    """Join connector-sized pieces on VM storage, validating every byte digest."""
    folder,destination = Path(folder),Path(destination)
    key = 'transformer_updated_binary_fold00'
    if (folder/'training_states'/key/'state.pt').exists() or (folder/'bert_colab'/(key+'.json')).exists():
        return None  # A newer Colab continuation supersedes the imported snapshot.
    manifest = json.loads((folder/'transfer.json').read_text())
    expected = manifest['files']['state.pt']
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = destination.with_name(destination.name+'.assembling')
    combined, count = hashlib.sha256(),0
    with temporary.open('wb') as writer:
        for part in manifest['checkpoint_parts']:
            if Path(part['name']).name!=part['name']:
                raise ValueError('Invalid checkpoint piece name')
            digest,size = hashlib.sha256(),0
            with (folder/part['name']).open('rb') as reader:
                for block in iter(lambda:reader.read(8*1024*1024),b''):
                    writer.write(block);digest.update(block);combined.update(block)
                    size+=len(block);count+=len(block)
            if size!=part['bytes'] or digest.hexdigest()!=part['sha256']:
                raise ValueError(f'Checkpoint piece failed verification: {part["name"]}')
            print('Verified checkpoint piece:',part['name'],flush=True)
    if count!=expected['bytes'] or combined.hexdigest()!=expected['sha256']:
        raise ValueError('Assembled checkpoint failed verification')
    temporary.replace(destination)
    return destination


def copy_atomic(source, destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = destination.with_name(destination.name+'.uploading')
    # Opening the source first holds a consistent snapshot even if training
    # atomically replaces its checkpoint during the copy.
    with source.open('rb') as reader,temporary.open('wb') as writer:
        stamp = os.fstat(reader.fileno())
        shutil.copyfileobj(reader,writer,8*1024*1024)
    temporary.replace(destination)
    return stamp.st_ino,stamp.st_mtime_ns,stamp.st_size


def sync_states(local, persistent, reports, stop, errors):
    copied = {}
    while True:
        for source in sorted(local.glob('*/state.pt')):
            key = source.parent.name
            try:
                stat = source.stat()
                stamp = stat.st_ino,stat.st_mtime_ns,stat.st_size
                if copied.get(key)==stamp:
                    continue
                copied[key] = copy_atomic(source,persistent/key/'state.pt')
                print(f'Drive checkpoint saved: {key}',flush=True)
            except (OSError,ValueError) as error:
                errors.append(str(error))
                print(f'Checkpoint backup will retry: {error}',flush=True)
        # Completed fold reports supersede their large rolling optimizer states.
        for report in reports.glob('*fold*.json'):
            target = persistent/report.stem/'state.pt'
            if target.exists():
                try:
                    target.unlink()
                except OSError as error:
                    errors.append(str(error))
        if stop.is_set():
            return
        stop.wait(30)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive-root',default='/content/drive/MyDrive/CL2 Full Corpus BERT')
    parser.add_argument('--work',default='/content/cl2_full_bert')
    parser.add_argument('--initial-state',help='Assembled imported checkpoint on runtime-local storage')
    parser.add_argument('--from-pretrained',action='store_true',help='Initialize missing folds from pretrained BERT; still resume existing Colab checkpoints')
    args = parser.parse_args(argv)
    root,drive = Path(args.work),Path(args.drive_root)
    import torch
    from huggingface_hub import snapshot_download
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime before starting the training cell')
    manifest = json.loads((root/'data/classification/full/manifest.json').read_text())
    if manifest['examples']!=3085832:
        raise ValueError('Unexpected corpus size; refusing a partial-dataset run')
    local = root/'ckpt/colab_training'
    persistent = drive/'training_states'
    reports = drive/'bert_colab'
    local.mkdir(parents=True,exist_ok=True)
    reports.mkdir(parents=True,exist_ok=True)
    for source in persistent.glob('*/state.pt'):
        if not (reports/(source.parent.name+'.json')).exists():
            copy_atomic(source,local/source.parent.name/'state.pt')
    initial = Path(args.initial_state) if args.initial_state else drive/'state.pt'
    first = local/'transformer_updated_binary_fold00/state.pt'
    if not first.exists() and not (reports/'transformer_updated_binary_fold00.json').exists():
        if not initial.exists() and not args.from_pretrained:
            raise FileNotFoundError('The transferred Mac checkpoint state.pt is missing')
        if initial.exists() and not args.from_pretrained:
            copy_atomic(initial,first)
    encoder = snapshot_download('bert-base-uncased',revision='86b5e0934494bd15c9632b12f734a8a67f723594',
                                cache_dir=str(root/'.cache/hf/hub'),
                                allow_patterns=['*.json','model.safetensors','vocab.txt'])
    command = [sys.executable,str(root/'scripts/classify_embeddings.py'),
               '--embedding','transformer','--encoder',encoder,'--classifier','mlp',
               '--modes','updated','--tasks','binary','multiclass',
               '--fold-manifest','results/classification/full/settings.json',
               '--epochs','2','--batch-size','128','--hidden','64','--head-lr','.001',
               '--encoder-lr','.00002','--device','cuda','--amp','--threads','4',
               '--out',str(reports),'--training-states',str(local),
               '--checkpoint-every','1000','--progress-every','100',
               '--save-last-fold','--checkpoints',str(drive/'models'),'--resume','--portable-resume']
    stop,errors = threading.Event(),[]
    worker = threading.Thread(target=sync_states,args=(local,persistent,reports,stop,errors),daemon=True)
    worker.start()
    record = {'device':torch.cuda.get_device_name(0),'torch':torch.__version__,
              'examples':manifest['examples'],'source_sha256':manifest['source_sha256'],
              'command':command,'started_unix':time.time(),'phase':'starting'}
    (reports/'colab_launch.json').write_text(json.dumps(record,indent=2)+'\n')
    print('GPU:',record['device'],'| Full-corpus examples:',manifest['examples'],flush=True)
    child = None
    try:
        child = subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,bufsize=1)
        with (drive/'bert_colab.log').open('a',encoding='utf8') as log:
            for line in child.stdout:
                print(line,end='',flush=True)
                log.write(line); log.flush()
        code = child.wait()
        record.update(phase='complete' if code==0 else 'failed',returncode=code)
        if code:
            raise RuntimeError(f'Training exited with code {code}; see bert_colab.log')
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.send_signal(2)
            child.wait()
        record['phase'] = 'interrupted; resume from the latest Drive checkpoint'
        raise
    finally:
        stop.set(); worker.join()
        record.update(updated_unix=time.time(),backup_errors=errors)
        (reports/'colab_launch.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
