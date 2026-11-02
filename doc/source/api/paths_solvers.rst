Path Solvers
============

A path solver computes the propagation :class:`~sionna.rt.Paths`
for a given :class:`~sionna.rt.Scene`.
This includes tracing the paths and computing the corresponding channel
coefficients, delays, and angles of departure and arrival.
Sionna provides a path solver (:class:`~sionna.rt.PathSolver`) which currently
supports specular reflections and diffuse reflections,
as well as refractions (or transmissions).

.. autoclass:: sionna.rt.PathSolver
   :members:
   :special-members: __call__

Resolving candidate hash collisions
----------------------------------

By default, candidate deduplication uses fixed-size hash counter arrays. Different
path hashes can map to the same array index, causing candidates to be discarded
even when the path output buffer is not full. With many receive antenna elements,
this can make a receiver's result depend on the other receivers in the solve.

An opt-in mode stores full 64-bit hashes and probes alternate slots when different
keys map to the same index:

.. code-block:: python

    solver = PathSolver(resolve_hash_collisions=True,
                        hash_table_size=4_000_000)
    paths = solver(scene, max_num_paths_per_src=1_000_000)

``hash_table_size`` specifies slots per hash table per source, independently of
the path output buffer. There are two tables, retaining the existing round/floor
geometric quantization checks. Their entries use eight bytes instead of four.
When omitted, the size is ``max(max_num_paths_per_src, 1000000)``.

This mode raises ``OverflowError`` if insertion cannot find a slot within 128
probes (or one table traversal for smaller tables), or if the per-source path
buffer overflows. Increase the indicated resource or solve fewer receivers per
call before accepting the result. A successful solve only rules out these two
forms of truncation: sampling, quantization, collisions of the full hashes, and
other physical/modeling approximations still require convergence checks.

Collision-resolving mode can use more memory and take longer. It does not provide
deterministic path ordering and currently cannot be combined with
``deterministic=True``. The default candidate generator is unchanged.

Copies of the same path
-----------------------

Candidates are deduplicated by a quantized fingerprint of the planes they hit.
Facets of one surface that are nearly, but not exactly, coplanar, as are common
in meshes built from CityGML or photogrammetry, get different fingerprints, and
the image method can resolve their candidates to the same path. Such copies
would add up in the channel coefficients.

The path solver therefore drops a path if another path has the same source,
target, and sequence of interactions: the same interaction type and shape at
every depth, the same primitive at specular reflections, and the same edge at
diffractions. Paths with a diffuse reflection are never dropped. Copies are
removed after the field computation, so they still occupy slots of
``max_num_paths_per_src``. Use ``PathSolver(discard_duplicate_paths=False)`` to
keep them.
