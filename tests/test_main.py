import pytest
from dataset2metadata.process import process


def test_braceexpand():
    try:
        process("./tests/ymls/test_braceexpand.yml")
        assert True
    except Exception as e:
        print(str(e))
        assert False


def test_custom_blip2():
    try:
        process("./examples/blip2/blip2.yml")
        assert True
    except Exception as e:
        print(str(e))
        assert False


def test_custom_blip2clipb32l14():
    try:
        process("./examples/blip2/blip2clipb32l14.yml")
        assert True
    except Exception as e:
        print(str(e))
        assert False


def test_local():
    try:
        process("./tests/ymls/test_local.yml")
        assert True
    except Exception as e:
        print(str(e))
        assert False


def test_s3():
    try:
        process("./tests/ymls/test_s3.yml")
        assert True
    except Exception as e:
        print(str(e))
        assert False


def test_cache():
    try:
        process("./tests/ymls/test_cache.yml")
        assert True
    except Exception as e:
        print(str(e))
        assert False


def test_datacomp_names():
    try:
        process("./tests/ymls/test_local_datacomp_names.yml")
        assert True
    except Exception as e:
        print(str(e))
        assert False
