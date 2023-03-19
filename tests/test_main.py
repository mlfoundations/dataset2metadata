import pytest
from dataset2metadata.process import process

def test_braceexpand():
    try:
        process('./tests/ymls/test_braceexpand.yml')
        assert True
    except:
        assert False

def test_custom():
    try:
        process('./tests/ymls/test_custom.yml')
        assert True
    except:
        assert False

def test_local():
    try:
        process('./tests/ymls/test_local.yml')
        assert True
    except:
        assert False

def test_s3():
    try:
        process('./tests/ymls/test_s3.yml')
        assert True
    except:
        assert False