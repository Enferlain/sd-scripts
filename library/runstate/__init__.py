"""The run state (D19, D21).

The owner pattern every owner of run state shares, generations and their
publication, the run authority with its records, transitions and
materialization, target references, and the projections consumers receive
instead of the authority. It imports only the IR system, observability fact
types and base, never the libraries, training or Torch: the types hold live
objects without importing Torch, and Torch-specific evidence and evaluators
live with the definitions that need them (D15).
"""
