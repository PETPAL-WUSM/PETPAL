import os
import numpy as np
import builtins
import types
import pytest
import pathlib

import petpal.preproc.regional_tac_extraction as rtx

class FakeLabelMapLoader:
    def __init__(self, label_map_option):
        self._label_map = {'R1': 1, 'R2': 2}
    @property
    def label_map(self):
        return self._label_map

class FakeImg:
    def __init__(self, arr):
        self._arr = arr
    def numpy(self):
        return self._arr

class FakeScanTiming:
    def __init__(self):
        self.start_in_mins = [0.0, 1.0, 2.0]
        self.end_in_mins = [1.0, 2.0, 3.0]
        self.center_in_mins = [0.5, 1.5, 2.5]

def fake_to_tsv(self, filename):
    # simple TSV writer for the TimeActivityCurve instances used in tests
    with open(filename, 'w') as fh:
        fh.write("time\tactivity\tuncertainty\n")
        for t, a, u in zip(self.times, self.activity, self.uncertainty):
            fh.write(f"{t}\t{a}\t{u}\n")

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Patch LabelMapLoader used in module
    monkeypatch.setattr(rtx, "LabelMapLoader", FakeLabelMapLoader)
    # Patch ants.image_read to return dummy arrays (not used because apply_mask_4d is patched)
    monkeypatch.setattr(rtx.ants, "image_read", lambda filename=None: FakeImg(np.zeros((2,2,2,3))))
    # Patch ScanTimingInfo.from_nifti
    monkeypatch.setattr(rtx.ScanTimingInfo, "from_nifti", lambda image_path=None: FakeScanTiming())
    # combine_regions_as_mask returns the label passed through so apply_mask_4d can distinguish
    monkeypatch.setattr(rtx, "combine_regions_as_mask", lambda segmentation_img, label: label)
    # Patch TimeActivityCurve.to_tsv to write simple TSV so tests can assert file creation
    monkeypatch.setattr(rtx.TimeActivityCurve, "to_tsv", fake_to_tsv)
    yield

def test_write_tacs_one_tsv_per_region_writes_only_non_nan_region(tmp_path, monkeypatch):
    # apply_mask_4d returns non-NaN voxels for label 1 and all-NaN voxels for label 2
    def fake_apply_mask_4d(input_arr, mask_arr, verbose=False):
        if mask_arr == 1:
            return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # mean is finite
        else:
            return np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])  # mean is nan
    monkeypatch.setattr(rtx, "apply_mask_4d", fake_apply_mask_4d)

    wr = rtx.WriteRegionalTacs(input_image_path="in.nii", segmentation_path="seg.nii", label_map="dummy")
    out_dir = tmp_path
    wr.write_tacs(out_tac_prefix="sub-01", out_tac_dir=str(out_dir), one_tsv_per_region=True)
    # Expect file for R1 only
    f_r1 = out_dir / "sub-01_seg-R1_tac.tsv"
    f_r2 = out_dir / "sub-01_seg-R2_tac.tsv"
    assert f_r1.exists()
    assert not f_r2.exists()
    # Basic content check
    content = f_r1.read_text()
    assert "time\tactivity\tuncertainty" in content
    assert "0.5\t" in content  # time present

def test_write_tacs_multitac_writes_combined_file_and_skips_nan_regions(tmp_path, monkeypatch):
    # same masking behavior as previous test
    def fake_apply_mask_4d(input_arr, mask_arr, verbose=False):
        if mask_arr == 1:
            return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        else:
            return np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    monkeypatch.setattr(rtx, "apply_mask_4d", fake_apply_mask_4d)

    wr = rtx.WriteRegionalTacs(input_image_path="in.nii", segmentation_path="seg.nii", label_map="dummy")
    out_dir = tmp_path
    wr.write_tacs(out_tac_prefix="sub-01", out_tac_dir=str(out_dir), one_tsv_per_region=False)
    combined = out_dir / "sub-01_multitacs.tsv"
    assert combined.exists()
    txt = combined.read_text()
    # Should contain frame_start(min) and R1 column but not R2 (R2 was NaN and skipped)
    assert "frame_start(min)" in txt
    assert "R1" in txt
    assert "R2" not in txt