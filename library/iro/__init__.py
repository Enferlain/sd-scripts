"""The IR system: Kuroiro on xDSL (D14, D16-D18, D21).

Kuroiro is the project's language for whole runs. A strategy is read into it,
normalized, verified and accepted; preparation and lowering turn the accepted
strategy into a running run. Its operations print under the ``iro`` prefix.

Section 13 builds the language half: the words, their operation contracts,
the contract registry, normalization, verification, the accepted run, the
written form, the reference semantics and the seams. Nothing here imports
Torch.
"""
