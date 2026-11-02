"""Diffraction into the shadow of wedges wider than 90 degrees, near and far
from the origin."""

import mitsuba as mi
import numpy as np
import pytest

from sionna import rt
from sionna.rt import PathSolver, Receiver, Transmitter, PlanarArray, \
    SceneObject, ITURadioMaterial
from sionna.rt.constants import InteractionType

# Box whose top rear edge, along y at x = BOX_DEPTH/2 and z = BOX_HEIGHT, has
# the tested interior angle: the rear face leans out toward the bottom
BOX_DEPTH, BOX_WIDTH, BOX_HEIGHT = 6.0, 10.0, 6.0
TX_POSITION = [-20.0, 0.0, 8.0]
# Below the line from the transmitter over the rear edge: in its shadow
RX_POSITIONS = [[20.0, -2.0, 1.5], [30.0, 0.0, 1.5], [40.0, 2.0, 3.0]]
SOLVER_KWARGS = {"max_depth": 1, "samples_per_src": int(1e6), "los": True,
                 "specular_reflection": False, "diffuse_reflection": False,
                 "refraction": False, "diffraction": True, "seed": 1}


def make_box_mesh(interior_angle_deg, origin):
    """Box with the rear face leaning so that its top rear edge has the given
    interior angle, translated by `origin`"""
    lean = BOX_HEIGHT*np.tan(np.radians(interior_angle_deg - 90.0))
    x_front, x_rear = -BOX_DEPTH/2, BOX_DEPTH/2
    y_half = BOX_WIDTH/2
    vertices = np.array([
        [x_front, -y_half, 0.], [x_rear + lean, -y_half, 0.],
        [x_rear + lean, y_half, 0.], [x_front, y_half, 0.],
        [x_front, -y_half, BOX_HEIGHT], [x_rear, -y_half, BOX_HEIGHT],
        [x_rear, y_half, BOX_HEIGHT], [x_front, y_half, BOX_HEIGHT],
    ]) + np.array(origin)
    # Consistently wound so that each edge is shared by two faces
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    mesh = mi.Mesh("box", len(vertices), len(faces))
    params = mi.traverse(mesh)
    params["vertex_positions"] = vertices.astype(np.float32).ravel()
    params["faces"] = faces.astype(np.uint32).ravel()
    params.update()
    return mesh


def count_diffracted_paths(interior_angle_deg, origin, deterministic):
    """Number of valid diffracted paths reaching each receiver"""
    scene = rt.load_scene()
    scene.frequency = 7e9
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                 polarization="V")
    scene.rx_array = scene.tx_array
    scene.edit(add=SceneObject(
        mi_mesh=make_box_mesh(interior_angle_deg, origin), name="box",
        radio_material=ITURadioMaterial("box-metal", "metal", thickness=0.01)))
    scene.add(Transmitter("tx", position=np.add(TX_POSITION, origin)))
    for i, rx_position in enumerate(RX_POSITIONS):
        scene.add(Receiver(f"rx-{i}", position=np.add(rx_position, origin)))

    paths = PathSolver(deterministic=deterministic)(scene, **SOLVER_KWARGS)
    valid = paths.valid.numpy()[:, 0, :]
    diffraction = paths.interactions.numpy()[0, :, 0, :] \
        == InteractionType.DIFFRACTION
    return (valid & diffraction).sum(axis=-1)


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("origin", [(0., 0., 0.), (400., 300., 0.)])
@pytest.mark.parametrize("interior_angle_deg", [80., 120., 150.])
def test_wide_wedge_diffraction(interior_angle_deg, origin, deterministic):
    """Each receiver in the shadow gets exactly one path over the rear edge"""
    counts = count_diffracted_paths(interior_angle_deg, origin, deterministic)
    assert counts.tolist() == [1]*len(RX_POSITIONS)
