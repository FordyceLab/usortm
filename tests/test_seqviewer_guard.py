"""Tests for failing early when the installed seqviewer is the wrong one.

The pileups are seqviewer's, so the two packages have to agree on what is
passed between them. An editable checkout shadows the pin in pyproject.toml
and can drift from it without anything saying so; the disagreement then
surfaces as a TypeError from the first pileup, which on a real run is more
than an hour in, after every expensive stage has already been paid for.
"""

import dataclasses
import sys
import types

import pytest

from usortm.demux.deps import DependencyError, check_seqviewer


def _fake_seqviewer(*, fields=("name", "ref_seq", "rows", "parent"),
                    names=("PileupGroup", "PileupView", "Read",
                           "grid_from_reads", "reads_from_alignment",
                           "render")):
    """A stand-in module exposing a chosen subset of the real surface."""
    module = types.ModuleType("seqviewer")
    module.__file__ = "/fake/seqviewer/__init__.py"

    group = dataclasses.make_dataclass(
        "PileupGroup", [(f, str, dataclasses.field(default="")) for f in fields]
    )
    for name in names:
        setattr(module, name, group if name == "PileupGroup" else object())
    return module


@pytest.fixture
def installed(monkeypatch):
    def install(module):
        monkeypatch.setitem(sys.modules, "seqviewer", module)
    return install


class TestWhenItPasses:

    def test_the_real_installed_package(self):
        """Whatever is installed here must satisfy what this package calls."""
        check_seqviewer()

    def test_a_module_with_everything(self, installed):
        installed(_fake_seqviewer())
        check_seqviewer()

    def test_extra_fields_are_fine(self, installed):
        installed(_fake_seqviewer(
            fields=("name", "ref_seq", "rows", "parent", "something_new")))
        check_seqviewer()


class TestWhenItFails:

    def test_the_rename_that_caused_this(self, installed):
        """PileupGroup.wild_type became .parent; a checkout on the wrong side
        of that is exactly what broke a run mid-flight."""
        installed(_fake_seqviewer(fields=("name", "ref_seq", "rows",
                                          "wild_type")))
        with pytest.raises(DependencyError, match=r"PileupGroup\.parent"):
            check_seqviewer()

    def test_a_missing_function(self, installed):
        installed(_fake_seqviewer(
            names=("PileupGroup", "PileupView", "Read", "render")))
        with pytest.raises(DependencyError, match="grid_from_reads"):
            check_seqviewer()

    def test_the_message_names_where_it_loaded_from(self, installed):
        """So the reader knows which checkout to fix."""
        installed(_fake_seqviewer(fields=("name", "ref_seq", "rows")))
        with pytest.raises(DependencyError, match="/fake/seqviewer"):
            check_seqviewer()

    def test_the_message_explains_the_editable_install(self, installed):
        installed(_fake_seqviewer(fields=("name", "ref_seq", "rows")))
        with pytest.raises(DependencyError, match="editable"):
            check_seqviewer()

    def test_not_installed_at_all(self, monkeypatch):
        import importlib

        real = importlib.import_module

        def refuse(name, *args, **kwargs):
            if name == "seqviewer":
                raise ImportError("no seqviewer")
            return real(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "seqviewer", raising=False)
        monkeypatch.setattr(importlib, "import_module", refuse)
        with pytest.raises(DependencyError, match="not installed"):
            check_seqviewer()


class TestItRunsBeforeTheExpensiveWork:

    def test_the_dependency_check_consults_it(self, monkeypatch):
        """It has to run at startup, or it saves nobody any time."""
        from usortm.demux import deps

        called = []
        monkeypatch.setattr(deps, "check_seqviewer",
                            lambda: called.append(True))
        monkeypatch.setattr(deps, "find_dorado", lambda: "/bin/dorado")
        monkeypatch.setattr(deps, "find_minimap2", lambda: "/bin/minimap2")
        monkeypatch.setattr(deps, "find_samtools", lambda: "/bin/samtools")
        deps.check_all_dependencies()
        assert called == [True]
