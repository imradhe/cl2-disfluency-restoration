import importlib.util
import hashlib
import json
import threading
from pathlib import Path


spec = importlib.util.spec_from_file_location('colab_resume',Path(__file__).resolve().parents[1]/'scripts/colab_resume.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_atomic_checkpoint_backup_and_completed_fold_cleanup(tmp_path):
    local,persistent,reports = [tmp_path/name for name in ['local','drive','reports']]
    reports.mkdir()
    source = local/'fold00/state.pt'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'initial checkpoint')
    target = persistent/'fold00/state.pt'
    module.copy_atomic(source,target)
    assert target.read_bytes()==b'initial checkpoint'
    source.write_bytes(b'new complete checkpoint')
    stop = threading.Event();stop.set()
    module.sync_states(local,persistent,reports,stop,[])
    assert target.read_bytes()==b'new complete checkpoint'
    assert not target.with_name('state.pt.uploading').exists()
    (reports/'fold00.json').write_text('{}')
    module.sync_states(local,persistent,reports,stop,[])
    assert not target.exists()


def test_split_checkpoint_reassembly_validates_pieces_and_combined_digest(tmp_path):
    import pytest
    folder=tmp_path/'drive';folder.mkdir()
    original=b'complete portable optimizer and model checkpoint'
    parts=[]
    for index,part in enumerate([original[:12],original[12:]]):
        name=f'state.pt.part{index:02d}';(folder/name).write_bytes(part)
        parts.append({'name':name,'bytes':len(part),'sha256':hashlib.sha256(part).hexdigest()})
    (folder/'transfer.json').write_text(json.dumps({'checkpoint_parts':parts,'files':{'state.pt':
        {'bytes':len(original),'sha256':hashlib.sha256(original).hexdigest()}}}))
    target=tmp_path/'vm/state.pt'
    assert module.assemble_checkpoint(folder,target)==target
    assert target.read_bytes()==original
    (folder/parts[1]['name']).write_bytes(b'corrupt transfer')
    with pytest.raises(ValueError,match='failed verification'):
        module.assemble_checkpoint(folder,target)
    assert target.read_bytes()==original
