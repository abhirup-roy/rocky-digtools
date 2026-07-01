"""Validate generic shell macros like directory swapping."""

import os
from rocky_digtools.utils import cd


class TestCd:
    def test_cd_changes_dir(self, tmp_path):
        original = os.getcwd()
        target = str(tmp_path / "subdir")
        os.makedirs(target)
        with cd(target):
            assert os.getcwd() == target
        assert os.getcwd() == original

    def test_cd_expanduser(self):
        original = os.getcwd()
        with cd("~"):
            assert os.getcwd() == os.path.expanduser("~")
        assert os.getcwd() == original

    def test_cd_nested(self, tmp_path):
        original = os.getcwd()
        dir_a = str(tmp_path / "a")
        dir_b = str(tmp_path / "b")
        os.makedirs(dir_a)
        os.makedirs(dir_b)
        with cd(dir_a):
            assert os.getcwd() == dir_a
            with cd(dir_b):
                assert os.getcwd() == dir_b
            assert os.getcwd() == dir_a
        assert os.getcwd() == original
