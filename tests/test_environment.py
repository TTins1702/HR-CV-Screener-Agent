import sys


def test_python_is_at_least_3_12():
    assert sys.version_info >= (3, 12)


def test_core_dependencies_import():
    import langgraph.graph
    import pandas
    import pydantic
    import yaml

    assert pydantic.VERSION.startswith("2.")
    assert hasattr(langgraph.graph, "StateGraph")
    assert hasattr(yaml, "safe_load")
    assert pandas.__version__


def test_src_package_is_importable():
    import src

    assert src.__doc__ == "HR CV Screener Agent."
