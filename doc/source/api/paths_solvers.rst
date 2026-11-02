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
