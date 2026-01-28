import tempfile
import os
import pytest
import pandas as pd
from petpal.io.csv import CsvWriter

def test_coerce_to_csv():
    path_with_multiple_suffixes = 'file.suffix1.suffix2'
    path_expected = os.path.join(os.getcwd(),'file.csv')
    assert CsvWriter.outpath_as_csv(path=path_with_multiple_suffixes)==path_expected

def test_writer_writes_csv():
    dataframe = pd.DataFrame(data={'a': 1},index=[0])
    writer = CsvWriter()
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_petpal_", dir='/tmp', suffix=".csv")
    os.close(fd)
    writer.write(df=dataframe, path=tmp_path)
    assert os.path.exists(tmp_path)
    print(tmp_path)
    with open(tmp_path) as f:
        assert f.read()==",a\n0,1\n"