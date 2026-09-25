"""Repository-wide pytest configuration.

``particlesim.viz.pytest_plugin`` adds ``--dashboard PATH``, which writes an
HTML dashboard of the benchmark tests a run executed (issue #83). ``pytester``
is pytest's own fixture for running pytest inside a test, which that plugin's
tests use.
"""

pytest_plugins = ("particlesim.viz.pytest_plugin", "pytester")
