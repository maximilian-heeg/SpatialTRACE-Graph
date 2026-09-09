"""Paper rendering validates omitted model identities without fake checkpoint files."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


def test_omitted_checkpoint_identity_is_strict(tmp_path, monkeypatch):
    paths = ModuleType('paper_paths')
    paths.ROOT, paths.Path = tmp_path, Path
    monkeypatch.setitem(sys.modules, 'paper_paths', paths)
    module_path = Path(__file__).resolve().parents[1]/'reproduction/checkpoint_lineage.py'
    spec = importlib.util.spec_from_file_location('tested_checkpoint_lineage', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    relative = 'assets/reference_graph.ckpt'
    record = {'path': relative, 'sha256': 'a'*64, 'bytes': 100}
    (tmp_path/'graph_checkpoint_lineage.json').write_text(json.dumps(
        {'binaries_distributed': False, 'checkpoints': [record]}))
    result = module.validate_graph_checkpoint_reference(tmp_path/relative, 'a'*64)
    assert 'identity_only' in result['verification_scope']
    assert not (tmp_path/relative).exists()
    with pytest.raises(ValueError, match='identity differs'):
        module.validate_graph_checkpoint_reference(tmp_path/relative, 'b'*64)
    with pytest.raises(ValueError, match='identity differs'):
        module.validate_graph_checkpoint_reference(tmp_path/'unlisted.ckpt', 'a'*64)
