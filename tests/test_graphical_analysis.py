import os
import warnings
import pytest
import pandas as pd

from petpal.utils.image_io import flatten_metadata
import petpal.kinetic_modeling.graphical_analysis as ga

def _make_instance(rsquared):
    inst = ga.MultiTACGraphicalAnalysis.__new__(ga.MultiTACGraphicalAnalysis)
    inst.analysis_props = [{'RSquared': rsquared}]
    inst.output_directory = "/tmp"
    inst.output_filename_prefix = "prefix"
    inst.method = "patlak"
    inst.inferred_seg_labels = ["roi1", "roi2"]
    return inst


def test_save_analysis_raises_if_run_not_called():
    inst = _make_instance(rsquared=None)
    with pytest.raises(RuntimeError):
        inst.save_analysis()

def test_save_analysis_calls_tsv_and_json(monkeypatch):
    inst = _make_instance(rsquared=0.95)
    calls = {"tsv": 0, "json": 0}

    def km_multifit_analysis_to_tsv_without_save_file(analysis_props: list[dict],
                                    output_directory: str,
                                    output_filename_prefix: str,
                                    method: str,
                                    inferred_seg_labels: list[str]):
        filename = f'{output_filename_prefix}_desc-{method}_fitprops.tsv'
        filepath = os.path.join(output_directory, filename)
        fit_table = pd.DataFrame()
        for seg_name, fit_props in zip(inferred_seg_labels, analysis_props):
            tmp_table = pd.DataFrame(flatten_metadata(fit_props),index=[seg_name])
            fit_table = pd.concat([fit_table,tmp_table])
        calls["tsv"] += 1
        assert isinstance(analysis_props, list)
        assert isinstance(output_directory, str)
        assert isinstance(output_filename_prefix, str)
        assert isinstance(method, str)
        assert isinstance(inferred_seg_labels, list)

    def km_multifit_analysis_to_jsons_without_save_file(analysis_props: list[dict],
                                    output_directory: str,
                                    output_filename_prefix: str,
                                    method: str,
                                    inferred_seg_labels: list[str]):
        for seg_name, fit_props in zip(inferred_seg_labels, analysis_props):
            filename = [output_filename_prefix,
                        f'desc-{method}',
                        f'seg-{seg_name}',
                        'fitprops.json']
            filename = '_'.join(filename)
            filepath = os.path.join(output_directory, filename)
        calls["json"] += 1
        assert isinstance(analysis_props, list)
        assert isinstance(output_directory, str)
        assert isinstance(output_filename_prefix, str)
        assert isinstance(method, str)
        assert isinstance(inferred_seg_labels, list)

    monkeypatch.setattr(ga, "km_multifit_analysis_to_tsv", km_multifit_analysis_to_tsv_without_save_file)
    monkeypatch.setattr(ga, "km_multifit_analysis_to_jsons", km_multifit_analysis_to_jsons_without_save_file)

    # Default behavior: TSV only
    calls["tsv"] = calls["json"] = 0
    inst.save_analysis(output_as_tsv=True, output_as_json=False)
    assert calls["tsv"] == 1 and calls["json"] == 0

    # JSON only
    calls["tsv"] = calls["json"] = 0
    inst.save_analysis(output_as_tsv=False, output_as_json=True)
    assert calls["tsv"] == 0 and calls["json"] == 1

    # Both
    calls["tsv"] = calls["json"] = 0
    inst.save_analysis(output_as_tsv=True, output_as_json=True)
    assert calls["tsv"] == 1 and calls["json"] == 1

def test_save_analysis_warns_when_no_output_requested(monkeypatch):
    inst = _make_instance(rsquared=0.5)
    # prevent actual functions being called if mistakenly invoked
    monkeypatch.setattr(ga, "km_multifit_analysis_to_tsv", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")))
    monkeypatch.setattr(ga, "km_multifit_analysis_to_jsons", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")))

    with pytest.warns(UserWarning):
        inst.save_analysis(output_as_tsv=False, output_as_json=False)