import pytest
from pydantic import ValidationError

from src.contracts.ablations import Ablations


def test_everything_is_on_by_default():
    ablations = Ablations()

    assert ablations.must_have_gate is True
    assert ablations.guard is True
    assert ablations.gray_zone is True
    assert ablations.label == "shipped"


def test_a_label_names_what_is_off_so_filenames_are_self_describing():
    assert Ablations(must_have_gate=False).label == "no_must_have_gate"
    assert Ablations(guard=False).label == "no_guard"
    assert Ablations(gray_zone=False).label == "no_gray_zone"
    assert (
        Ablations(guard=False, gray_zone=False).label
        == "no_guard+no_gray_zone"
    )


def test_a_run_configuration_cannot_be_mutated_halfway_through():
    ablations = Ablations()

    with pytest.raises(ValidationError):
        ablations.must_have_gate = False
