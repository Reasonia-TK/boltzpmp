from __future__ import annotations

from importlib.metadata import distribution

import boltzpmp
from boltzpmp import _core


def test_distribution_import_and_extension_names_match() -> None:
    assert distribution("boltzpmp").metadata["Name"] == "boltzpmp"
    assert boltzpmp.__name__ == "boltzpmp"
    assert _core.__name__ == "boltzpmp._core"
