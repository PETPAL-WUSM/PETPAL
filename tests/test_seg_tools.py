import pytest
from unittest.mock import Mock
from petpal.preproc import segmentation_tools as st

def test_seg_crop_to_pet_fov_masks_and_returns_mask(monkeypatch):
    pet_img = object()
    seg_img = object()
    fake_mean = object()
    fake_mask = object()
    fake_result = object()

    m_check = Mock(return_value=True)
    m_get_avg = Mock(return_value=fake_mean)
    m_thresh = Mock(return_value=fake_mask)
    m_mask = Mock(return_value=fake_result)

    monkeypatch.setattr(st, "check_physical_space_for_ants_image_pair", m_check)
    monkeypatch.setattr(st, "get_average_of_timeseries", m_get_avg)
    monkeypatch.setattr(st, "ants", Mock())
    st.ants.threshold_image = m_thresh
    st.ants.mask_image = m_mask

    res = st.seg_crop_to_pet_fov(pet_img=pet_img, segmentation_img=seg_img)

    assert res is fake_result
    m_check.assert_called_once_with(pet_img, seg_img)
    m_get_avg.assert_called_once_with(input_image=pet_img)
    m_thresh.assert_called_once_with(fake_mean, 1e-36)
    m_mask.assert_called_once_with(seg_img, fake_mask)

def test_seg_crop_to_pet_fov_asserts_on_physical_space(monkeypatch):
    pet_img = object()
    seg_img = object()
    monkeypatch.setattr(st, "check_physical_space_for_ants_image_pair", Mock(return_value=False))
    with pytest.raises(AssertionError):
        st.seg_crop_to_pet_fov(pet_img, seg_img)