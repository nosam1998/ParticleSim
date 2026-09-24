"""Adapters to established codes (design doc ADR-005).

Each translates a ParticleSim configuration into another code's input and
reads that code's output back, so a problem set up and checked here can be
run at a scale this package does not attempt. Nothing of the other code is
distributed: see ``docs/adapters/`` for each one's licence record.
"""
