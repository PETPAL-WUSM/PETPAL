import pytest
from petpal.preproc.symmetric_geometric_transfer_matrix import Sgtm

class DummyImage:
    def __init__(self, dimension):
        self.dimension = dimension

def make_sgtm_with_dimension(dim, sgtm_result=None):
    sgtm = object.__new__(Sgtm)
    sgtm.input_image = DummyImage(dimension=dim)
    sgtm.sgtm_result = sgtm_result
    return sgtm

def test_save_calls_save_results_3d_for_3d_image():
    sgtm = make_sgtm_with_dimension(3, sgtm_result=("labels", "vals", 1.0))
    called = {}
    def fake_save_results_3d(sgtm_result, out_tsv_path):
        called['args'] = (sgtm_result, out_tsv_path)
    sgtm.save_results_3d = fake_save_results_3d

    sgtm.save(output_path="out.tsv")

    assert 'args' in called
    assert called['args'][0] is sgtm.sgtm_result
    assert called['args'][1] == "out.tsv"

def test_save_calls_save_results_4d_tacs_when_one_tsv_per_region_true():
    sgtm = make_sgtm_with_dimension(4, sgtm_result="frame_results")
    called = {}
    def fake_save_results_4d_tacs(sgtm_result, out_tac_dir, out_tac_prefix):
        called['args'] = (sgtm_result, out_tac_dir, out_tac_prefix)
    sgtm.save_results_4d_tacs = fake_save_results_4d_tacs

    sgtm.save(output_path="/tmp/dir", out_tac_prefix="pref", one_tsv_per_region=True)

    assert 'args' in called
    assert called['args'][0] is sgtm.sgtm_result
    assert called['args'][1] == "/tmp/dir"
    assert called['args'][2] == "pref"

def test_save_calls_save_results_4d_multitacs_when_one_tsv_per_region_false():
    sgtm = make_sgtm_with_dimension(4, sgtm_result="frame_results")
    called = {}
    def fake_save_results_4d_multitacs(sgtm_result, out_tac_dir, out_tac_prefix):
        called['args'] = (sgtm_result, out_tac_dir, out_tac_prefix)
    sgtm.save_results_4d_multitacs = fake_save_results_4d_multitacs

    sgtm.save(output_path="/tmp/dir2", out_tac_prefix="pref2", one_tsv_per_region=False)

    assert 'args' in called
    assert called['args'][0] is sgtm.sgtm_result
    assert called['args'][1] == "/tmp/dir2"
    assert called['args'][2] == "pref2"