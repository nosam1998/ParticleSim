"""Worked hypothesis templates (design doc Section 8, Appendix B).

Each module here is a complete, runnable theory plugin in one of the four
forms a singularity hypothesis can take. They are importable and exercised by
the test suite, so a template that stops working fails the build rather than
misleading the next person who copies it.

They are deliberately not registered as entry points: they are starting
points to copy, not theories anyone should find in ``particlesim theories``.
"""
