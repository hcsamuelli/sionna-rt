#
# SPDX-FileCopyrightText: Copyright (c) 2026 The Sionna contributors.
# SPDX-License-Identifier: Apache-2.0
#
"""Copies of the same path are dropped by the path solver."""

import mitsuba as mi
import numpy as np
import pytest

from sionna import rt
from sionna.rt import PathSolver, Receiver, Transmitter, PlanarArray, \
    SceneObject, ITURadioMaterial

# A wall made of facets tilted against each other by more than the candidate
# generator's plane quantization, so that every facet yields its own candidate
WALL_X, WALL_WIDTH, WALL_HEIGHT, FACET_TILT = 120.0, 40.0, 20.0, 7e-5
TX_POSITION = [WALL_X - 25.0, 0.0, 8.0]
RX_POSITIONS = [[WALL_X - distance, y, 1.5] for distance in (10.0, 20.0, 30.0)
                for y in (-12.0, -4.0, 4.0, 12.0)]
SOLVER_KWARGS = {"max_depth": 1, "samples_per_src": int(1e6), "los": False,
                 "specular_reflection": True, "diffuse_reflection": False,
                 "refraction": False, "diffraction": False, "seed": 42}


def make_wall_quad(y_start, y_end, z_start, z_end):
    """Corners of an upright rectangle in the wall plane"""
    return np.array([[WALL_X, y_start, z_start], [WALL_X, y_end, z_start],
                     [WALL_X, y_end, z_end], [WALL_X, y_start, z_end]])


def rotate_corners(corners, angle_z, angle_y, center):
    """Corners rotated about `center`, first about z, then about y"""
    rotation_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                           [np.sin(angle_z), np.cos(angle_z), 0],
                           [0, 0, 1]])
    rotation_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                           [0, 1, 0],
                           [-np.sin(angle_y), 0, np.cos(angle_y)]])
    return (corners - center) @ (rotation_y @ rotation_z).T + center


def make_test_walls():
    """Wall name -> quads: two tilted halves, or 800 randomly tilted tiles"""
    rng = np.random.default_rng(0)
    two_halves = [make_wall_quad(-WALL_WIDTH/2, 0, 0, WALL_HEIGHT),
                  rotate_corners(
                      make_wall_quad(0, WALL_WIDTH/2, 0, WALL_HEIGHT),
                      FACET_TILT, 0, [WALL_X, 0, 0])]
    tiles = [rotate_corners(tile, *FACET_TILT*rng.normal(size=2)/np.sqrt(2),
                            tile.mean(0))
             for y in range(-20, 20) for z in range(20)
             for tile in [make_wall_quad(y, y + 1, z, z + 1)]]
    return {"two-halves": two_halves, "800-tiles": tiles}


def make_flat_wall():
    """The wall as one flat quad"""
    return [make_wall_quad(-WALL_WIDTH/2, WALL_WIDTH/2, 0, WALL_HEIGHT)]


def build_quad_mesh(name, quads):
    """Mitsuba mesh of two triangles per quad"""
    vertices = np.vstack(quads)
    faces = (np.array([[0, 1, 2], [0, 2, 3]])
             + 4*np.arange(len(quads))[:, None, None]).reshape(-1, 3)
    mesh = mi.Mesh(name, len(vertices), len(faces))
    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = vertices.ravel().tolist()
    mesh_params["faces"] = faces.ravel().tolist()
    mesh_params.update()
    return mesh


def build_wall_scene(quads):
    """Empty scene with the concrete wall, the transmitter and the receivers"""
    scene = rt.load_scene()
    scene.frequency = 7e9
    scene.add(SceneObject(mi_mesh=build_quad_mesh("wall", quads), name="wall",
                          radio_material=ITURadioMaterial(
                              "concrete", "concrete", thickness=0.3)))
    scene.tx_array = scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(Transmitter("tx", position=TX_POSITION))
    for i, rx_position in enumerate(RX_POSITIONS):
        scene.add(Receiver(f"rx{i}", position=rx_position))
    return scene


def collect_paths_per_receiver(paths):
    """Set of (coefficient, delay) of the valid paths of every receiver"""
    num_receivers = len(RX_POSITIONS)
    valid = paths.valid.numpy().reshape(num_receivers, -1)
    coefficients = (paths.a[0].numpy()
                    + 1j*paths.a[1].numpy()).reshape(num_receivers, -1)
    delays = paths.tau.numpy().reshape(num_receivers, -1)
    return [{(complex(coefficients[rx, i]), float(delays[rx, i]))
             for i in np.flatnonzero(valid[rx])} for rx in range(num_receivers)]


def count_paths_per_receiver(paths, interaction_type, depth=0):
    """Valid paths of every receiver with `interaction_type` at `depth`"""
    has_interaction = paths.interactions.numpy()[depth] == interaction_type
    return np.sum(has_interaction & paths.valid.numpy(), axis=-1).reshape(-1)


@pytest.mark.parametrize("wall_name", list(make_test_walls().keys()))
@pytest.mark.parametrize("solver_kwargs", [
    pytest.param({}, id="default"),
    pytest.param({"deterministic": True}, id="deterministic"),
])
def test_one_reflection_per_receiver(wall_name, solver_kwargs):
    scene = build_wall_scene(make_test_walls()[wall_name])
    kept_paths = PathSolver(**solver_kwargs)(scene, **SOLVER_KWARGS)
    all_paths = PathSolver(discard_duplicate_paths=False,
                           **solver_kwargs)(scene, **SOLVER_KWARGS)

    # One reflection per receiver, where the solver would otherwise give copies
    num_receivers = len(RX_POSITIONS)
    num_kept = kept_paths.valid.numpy().reshape(num_receivers, -1).sum(-1)
    num_all = all_paths.valid.numpy().reshape(num_receivers, -1).sum(-1)
    assert (num_kept == 1).all()
    assert (num_all >= 1).all() and (num_all > 1).any()

    # The kept paths are unchanged by dropping the copies
    if solver_kwargs.get("deterministic"):
        assert all(kept <= every for kept, every in zip(
            collect_paths_per_receiver(kept_paths),
            collect_paths_per_receiver(all_paths)))


def test_diffuse_paths_are_never_dropped():
    scene = build_wall_scene(make_test_walls()["two-halves"])
    scene.get("wall").radio_material.scattering_coefficient = 0.5
    solver_kwargs = {**SOLVER_KWARGS, "diffuse_reflection": True,
                     "samples_per_src": 10000}
    kept_paths = PathSolver()(scene, **solver_kwargs)
    all_paths = PathSolver(discard_duplicate_paths=False)(scene,
                                                          **solver_kwargs)

    diffuse = rt.InteractionType.DIFFUSE
    num_kept = count_paths_per_receiver(kept_paths, diffuse).sum()
    num_all = count_paths_per_receiver(all_paths, diffuse).sum()
    assert num_kept == num_all > 0


@pytest.mark.parametrize("wall_name", ["flat", "two-halves"])
def test_diffraction_paths_are_kept(wall_name):
    """Distinct diffraction paths stay, copies of reflections go"""
    quads = make_flat_wall() if wall_name == "flat" \
        else make_test_walls()["two-halves"]
    scene = build_wall_scene(quads)
    solver_kwargs = {**SOLVER_KWARGS, "diffraction": True,
                     "edge_diffraction": True}
    kept_paths = PathSolver()(scene, **solver_kwargs)
    all_paths = PathSolver(discard_duplicate_paths=False)(scene,
                                                          **solver_kwargs)

    # The seam between the two halves gives edges of its own, which stay
    specular, diffraction = rt.InteractionType.SPECULAR, \
        rt.InteractionType.DIFFRACTION
    assert (count_paths_per_receiver(kept_paths, specular) == 1).all()
    assert (count_paths_per_receiver(kept_paths, diffraction)
            == count_paths_per_receiver(all_paths, diffraction)).all()
    assert (count_paths_per_receiver(kept_paths, diffraction) > 0).all()


@pytest.mark.parametrize("deterministic", [False, True])
def test_copies_of_reflection_then_diffraction_are_dropped(deterministic):
    """Copies that also bend over the top edge of a screen are dropped, so the
    tilted wall gives the same paths as the flat one"""
    path_counts = {}
    for wall_name in ("flat", "two-halves"):
        quads = make_flat_wall() if wall_name == "flat" \
            else make_test_walls()["two-halves"]
        scene = build_wall_scene(quads)

        # A screen between the wall and new receivers behind it
        screen = np.array([[105.0, -30.0, 0.0], [105.0, 30.0, 0.0],
                           [105.0, 30.0, 6.0], [105.0, -30.0, 6.0]])
        scene.add(SceneObject(mi_mesh=build_quad_mesh("screen", [screen]),
                              name="screen",
                              radio_material=ITURadioMaterial(
                                  "metal", "metal", thickness=0.01)))
        for i in range(len(RX_POSITIONS)):
            scene.remove(f"rx{i}")
        scene.remove("tx")
        scene.add(Transmitter("tx", position=[95.0, 0.0, 8.0]))
        for i, y in enumerate((-12.0, -4.0, 4.0, 12.0)):
            scene.add(Receiver(f"rx{i}", position=[100.0, y, 1.5]))

        # Reflection then diffraction, and diffraction only, with and without
        # dropping copies
        solver_kwargs = {**SOLVER_KWARGS, "max_depth": 2, "diffraction": True,
                         "edge_diffraction": True}
        for result_name, discard in (("kept", True), ("all", False)):
            paths = PathSolver(deterministic=deterministic,
                               discard_duplicate_paths=discard)(scene,
                                                                **solver_kwargs)
            interaction_types = paths.interactions.numpy().reshape(2, -1)
            valid = paths.valid.numpy().reshape(-1)
            is_reflection_then_diffraction = valid \
                & (interaction_types[0] == rt.InteractionType.SPECULAR) \
                & (interaction_types[1] == rt.InteractionType.DIFFRACTION)
            is_diffraction_only = valid \
                & (interaction_types[0] == rt.InteractionType.DIFFRACTION)
            path_counts[wall_name, result_name] = (
                int(is_reflection_then_diffraction.sum()),
                int(is_diffraction_only.sum()))

    assert path_counts["flat", "kept"] == path_counts["flat", "all"]
    assert path_counts["two-halves", "kept"] == path_counts["flat", "kept"]
    assert path_counts["two-halves", "all"][0] \
        > path_counts["two-halves", "kept"][0] > 0
    assert path_counts["two-halves", "all"][1] \
        == path_counts["two-halves", "kept"][1] > 0
