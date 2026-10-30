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
    def fake_save_results_3d(sgtm_result, out_sgtm_path):
        called['args'] = (sgtm_result, out_sgtm_path)
    sgtm.save_results_3d = fake_save_results_3d

    sgtm.save(out_sgtm_path="out.tsv")

    assert 'args' in called
    assert called['args'][0] is sgtm.sgtm_result
    assert called['args'][1] == "out.tsv"


def test_save_calls_save_results_4d_multitacs():
    sgtm = make_sgtm_with_dimension(4, sgtm_result="frame_results")
    called = {}
    def fake_save_results_4d_multitacs(sgtm_result, out_sgtm_path):
        called['args'] = (sgtm_result, out_sgtm_path)
    sgtm.save_results_4d_multitacs = fake_save_results_4d_multitacs

    sgtm.save(out_sgtm_path="/tmp/dir2/pref2.tsv")

    assert 'args' in called
    assert called['args'][0] is sgtm.sgtm_result
    assert called['args'][1] == "/tmp/dir2/pref2.tsv"