import numpy as np
import pytest
import torch
from spatial_axis_gat.graph import build_batch_spatial_edge_index
from spatial_axis_gat.model import GATModel
from spatial_axis_gat.workflow import Neighborhoods, predict_nodes, create_demo, read_input
from spatial_axis_gat.cli import main


def test_exact_minibatches_match_whole_graph():
    torch.manual_seed(19)
    rng = np.random.default_rng(19)
    coords = rng.normal(size=(45, 2))
    sections = np.repeat(['a', 'b', 'c'], 15)
    edges = build_batch_spatial_edge_index(coords, sections, n_neighbors=5)
    model = GATModel(8, 16, n_heads=2, dropout=0).eval()
    x = torch.randn(45, 8)
    with torch.inference_mode():
        expected = model(x, edges).numpy()
    actual = predict_nodes(model, x, Neighborhoods(edges, len(x)), batch_size=7)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_singletons_and_two_cell_sections():
    edges = build_batch_spatial_edge_index(np.array([[0, 0], [1, 0], [9, 9]]),
        np.array(['pair', 'pair', 'alone']), n_neighbors=20)
    assert set(map(tuple, edges.T.tolist())) == {(0, 1), (1, 0)}


def test_train_predict_and_reject_wrong_features(tmp_path):
    data = tmp_path / 'demo.h5ad'
    create_demo(data)
    run = tmp_path / 'train'
    main(['train', '--input', str(data), '--target', 'crypt_villus', '--output-dir', str(run),
          '--epochs', '2', '--hidden-features', '8', '--heads', '2', '--batch-size', '16'])
    output = tmp_path / 'predictions.csv'
    args = ['predict', '--input', str(data), '--checkpoint', str(run / 'model.pt'), '--output', str(output)]
    main(args)
    assert output.exists() and (run / 'split.csv').exists()
    with pytest.raises(FileExistsError):
        main(args)
    adata, *_ = read_input(data)
    adata.uns['tissuemapper_feature_space'] = 'unrelated-scvi-fit'
    wrong = tmp_path / 'wrong.h5ad'
    adata.write_h5ad(wrong)
    args[2] = str(wrong)
    args[-1] = str(tmp_path/'incompatible.csv')
    with pytest.raises(ValueError, match='Incompatible feature space'):
        main(args)


def test_training_rejects_section_leakage(tmp_path):
    data = tmp_path / 'demo.h5ad'
    create_demo(data)
    adata, *_ = read_input(data)
    adata.obs['split'] = adata.obs['split'].astype(str)
    adata.obs.loc[adata.obs_names[0], 'split'] = 'test'
    bad = tmp_path / 'bad.h5ad'
    adata.write_h5ad(bad)
    with pytest.raises(ValueError, match='crosses supervised splits'):
        main(['train', '--input', str(bad), '--target', 'crypt_villus',
              '--output-dir', str(tmp_path / 'out')])


def test_classifier_train_predict(tmp_path):
    import pandas as pd
    data = tmp_path / 'demo.h5ad'
    create_demo(data)
    run = tmp_path / 'classifier'
    main(['train', '--input', str(data), '--target', 'peyer_label', '--task', 'peyer',
          '--output-dir', str(run), '--epochs', '2', '--hidden-features', '8', '--heads', '2'])
    payload = torch.load(run/'model.pt', weights_only=True)
    adata, x, _, sections = read_input(data)
    torch.testing.assert_close(payload['feature_mean'], x[sections == 'section_a'].mean(0))
    output = run/'predictions.csv'
    main(['predict', '--input', str(data), '--checkpoint', str(run/'model.pt'), '--output', str(output)])
    assert pd.read_csv(output).peyer_probability.between(0, 1).all()
    assert (run/'provenance.json').is_file()
    assert output.with_suffix('.provenance.json').is_file()
