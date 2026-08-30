"""Doctor output is the first thing anyone sees, so it gets tested."""

import warnings

import pytest

from videocortex_spark import doctor
from videocortex_spark.config import FEATURE_EXTRACTORS, HF_CHECKPOINT_PROBE
from videocortex_spark.doctor import FAIL, OK, UNKNOWN, WARN, Check


def test_every_repo_has_a_probe_file():
    """The gated check downloads this file; a missing entry silently skips it."""
    assert HF_CHECKPOINT_PROBE
    for name, spec in FEATURE_EXTRACTORS.items():
        assert spec.get("probe"), f"{name} has no probe file"


def test_offline_run_skips_network_checks_but_still_reports():
    checks = doctor.run_all(network=False)
    names = [c.name for c in checks]
    assert "python" in names
    assert "torch" in names
    assert not any(n.startswith("hf:") for n in names)
    assert "hf token" not in names


def test_renderer_run_skips_the_model_stack():
    checks = doctor.run_all(model=False)
    names = [c.name for c in checks]
    assert names == ["python", "render stack", "ffmpeg", "fsaverage5"]
    assert doctor.exit_code(checks) == 0


def test_exit_code_is_nonzero_only_on_failure():
    assert doctor.exit_code([Check("a", OK, "")]) == 0
    assert doctor.exit_code([Check("a", WARN, ""), Check("b", UNKNOWN, "")]) == 0
    assert doctor.exit_code([Check("a", OK, ""), Check("b", FAIL, "")]) == 1


def test_report_shows_the_fix_for_problems_only():
    report = doctor.format_report(
        [
            Check("broken", FAIL, "it is broken", fix="do the thing"),
            Check("fine", OK, "all good", fix="never shown"),
        ]
    )
    assert "do the thing" in report
    assert "never shown" not in report
    assert "1 blocking problem" in report


def test_report_is_clean_when_nothing_is_wrong():
    assert "All clear" in doctor.format_report([Check("a", OK, "yes")])


def test_python_check_matches_the_interpreter_running_the_tests():
    # tests only run on >=3.11, which is what tribev2 needs
    assert doctor.check_python().status == OK


@pytest.mark.skipif(not doctor._has("tribev2"), reason="predict extra not installed")
def test_tribev2_import_does_not_leak_neuralset_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        doctor.check_tribev2()
    leaked = [w for w in caught if "neuralset" in getattr(w, "filename", "")]
    assert leaked == []
