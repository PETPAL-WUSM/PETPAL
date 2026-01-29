import tempfile
import os
import pytest
import pandas as pd
from petpal.utils.useful_functions import coerce_outpath_extension
from petpal.io.table import TableSaver

def test_coerce_to_csv():
    path_with_multiple_suffixes = 'file.suffix1.suffix2'
    path_expected = os.path.join(os.getcwd(),'file.csv')
    assert coerce_outpath_extension(path=path_with_multiple_suffixes, ext='.csv')==path_expected

def test_saver_saves_table():
    dataframe = pd.DataFrame(data={'a': 1},index=[0])
    saver = TableSaver()
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_petpal_", dir='/tmp', suffix=".csv")
    os.close(fd)
    saver.save(df=dataframe, path=tmp_path)
    assert os.path.exists(tmp_path)
    print(tmp_path)
    with open(tmp_path) as f:
        assert f.read()==",a\n0,1\n"