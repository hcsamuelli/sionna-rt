#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Buffer to store paths during their computation"""

import mitsuba as mi
import drjit as dr
from typing import List, Union, Type, Tuple
import dataclasses

from sionna.rt.constants import InteractionType, INVALID_SHAPE,\
    INVALID_PRIMITIVE
from sionna.rt.utils import WedgeGeometry, hash_fnv1a


class PathsBufferBase:
    """
    Base class for paths buffer, containing only the fields that need to be
    read from & written to in the same symbolic loop.
    When needed, we will create a copy of this object to avoid write-after-read.

    :param buffer_size: Size of the buffer
    :param max_depth: Maximum depth
    """

    def __init__(self, buffer_size: int, max_depth: int):
        # Size of the array
        # If `max_depth` is 0, we allocate one element to store paths data
        depth_dim_size = max(1, max_depth)

        self._max_depth = max_depth
        self._buffer_size = buffer_size
        self._depth_dim_size = depth_dim_size

        # Path vertices
        # The additional two entries along the depth dimension correspond to the
        # endpoints of the paths.
        self._vertices_x = dr.zeros(mi.TensorXf, [buffer_size, depth_dim_size])
        self._vertices_y = dr.zeros(mi.TensorXf, [buffer_size, depth_dim_size])
        self._vertices_z = dr.zeros(mi.TensorXf, [buffer_size, depth_dim_size])

        # Pointers to intersected shapes
        # Shapes pointers are stored as unsigned integers
        self._shapes = dr.full(mi.TensorXu, INVALID_SHAPE,
                                   [buffer_size, depth_dim_size])

        # Index of intersected primitive of the shape
        self._primitives = dr.full(mi.TensorXu, INVALID_PRIMITIVE,
                                   [buffer_size, depth_dim_size])

        # Used for a workaround required (for now) on the CUDA backend
        self._tensor_width_lit = dr.width(self._shapes.array)
        self._tensor_width = dr.opaque(mi.UInt32, self._tensor_width_lit)


    @staticmethod
    def from_paths_buffer(paths: "PathsBuffer"):
        # pylint: disable=protected-access
        """
        Creates a new basic paths buffer from an existing paths buffer by
        copying the relevant fields.
        This is useful to avoid reading & writing to the same arrays within
        a symbolic loop.
        """
        result = PathsBufferBase(paths.buffer_size, paths.max_depth)
        # Note: lazy copies
        result._vertices_x = dr.copy(paths._vertices_x)
        result._vertices_y = dr.copy(paths._vertices_y)
        result._vertices_z = dr.copy(paths._vertices_z)
        result._shapes = dr.copy(paths._shapes)
        result._primitives = dr.copy(paths._primitives)
        return result


    @property
    def max_depth(self):
        r"""Maximum depth

        :type: :py:class:`int`
        """
        return self._max_depth

    @property
    def depth_dim_size(self):
        r"""Size of the depth dimension of the arrays and tensors instantiated
        by this class. Equals to 1 if ``max_depth == 0`` or to ``max_depth``
        otherwise.

        :type: :py:class:`int`
        """
        return self._depth_dim_size

    @property
    def buffer_size(self):
        r"""Size of the buffer

        :type: :py:class:`int`
        """
        return self._buffer_size

    @property
    def vertices_x(self):
        r"""X coordinates of the paths' vertices

        :type: :py:class:`mi.TensorXf [buffer_size, depth_dim_size]`
        """
        return self._vertices_x

    @property
    def vertices_y(self):
        r"""Y coordinates of the paths' vertices

        :type: :py:class:`mi.TensorXf [buffer_size, depth_dim_size]`
        """
        return self._vertices_y

    @property
    def vertices_z(self):
        r"""Z coordinates of the paths' vertices

        :type: :py:class:`mi.TensorXf [buffer_size, depth_dim_size]`
        """
        return self._vertices_z

    @property
    def shapes(self):
        r"""Intersected shapes. Invalid shapes are represented by
        :data:`~sionna.rt.constants.INVALID_SHAPE`

        For :data:`~sionna.rt.constants.InteractionType.SENSING` interactions,
        the index of the sensing target is stored instead of a shape.

        :type: :py:class:`mi.TensorXu [buffer_size, depth_dim_size]`
        """
        return self._shapes

    @property
    def primitives(self):
        r"""Intersected primitives. Invalid primitives are represented by
        :data:`~sionna.rt.constants.INVALID_PRIMITIVE`.

        For :data:`~sionna.rt.constants.InteractionType.SENSING` interactions,
        the index, within its sensing target, of the sensing point is stored
        instead of a primitive.

        :type: :py:class:`mi.TensorXu [buffer_size, depth_dim_size]`
        """
        return self._primitives

    def get_vertex(self, depth: mi.UInt, active: mi.Bool) -> mi.Point3f:
        r"""
        Gathers the coordinates of the vertices of the paths for the specified
        ``depth``

        :param depth: Depths
        :param active: Flags specifying active components

        :return: Coordinates of the paths vertices
        """

        vx = self._gather_depth(mi.Float, self._vertices_x, depth, active)
        vy = self._gather_depth(mi.Float, self._vertices_y, depth, active)
        vz = self._gather_depth(mi.Float, self._vertices_z, depth, active)

        return mi.Point3f(vx, vy, vz)

    def get_shape(self, depth: mi.UInt, active: mi.Bool) -> mi.UInt:
        r"""
        Gathers the indices to the intersected shapes for the specified
        ``depth``

        :param depth: Depths
        :param active: Flags specifying active components

        :return: Indices of the intersected shapes
        """

        return self._gather_depth(mi.UInt, self._shapes, depth, active)

    def get_primitive(self, depth: mi.UInt, active: mi.Bool) -> mi.UInt:
        r"""
        Gathers the indices to the intersected primitives for the specified
        ``depth``

        :param depth: Depths
        :param active: Flags specifying active components

        :return: Indices of the intersected primitives
        """

        return self._gather_depth(mi.UInt, self._primitives, depth, active)

    def get_primitive_props(self,
                            depth: mi.UInt,
                            return_normal: bool,
                            return_vertices: bool,
                            active: mi.Bool
        ) -> mi.Vector3f\
            | Tuple[mi.Point3f, mi.Point3f, mi.Point3f]\
            | Tuple[mi.Vector3f, mi.Point3f, mi.Point3f, mi.Point3f]:
        r"""
        Gathers primitive properties (normals and/or vertices) for the specified
        ``depth``

        :param depth: Depths
        :param return_normal: Whether to return face normals
        :param return_vertices: Whether to return vertex positions
        :param active: Flags specifying active components

        :return: Tuple containing requested primitive properties:
            - Face normal (if return_normal=True)
            - Vertex positions (if return_vertices=True)
        """

        valid = active & (depth > 0)

        # Gather shape pointer for the input depth and cast it to a mesh
        # pointer
        shape_int = self.get_shape(depth, valid)
        mesh_ptr = dr.reinterpret_array(mi.MeshPtr, shape_int)

        # Primitive index
        prim_ind = self.get_primitive(depth, valid)

        output = []
        # Normal
        if return_normal:
            normal = mesh_ptr.face_normal(prim_ind, valid)
            if not return_vertices:
                return normal
            output.append(normal)
        # Vertices
        if return_vertices:
            v_ind = mesh_ptr.face_indices(prim_ind, valid)
            output.append(mesh_ptr.vertex_position(v_ind.x, valid))
            output.append(mesh_ptr.vertex_position(v_ind.y, valid))
            output.append(mesh_ptr.vertex_position(v_ind.z, valid))
        return tuple(output)

    ###############################################
    # Internal methods
    ###############################################

    def _gather_depth(self,
                      dtype: Type[Union[mi.Float, mi.UInt, mi.Bool]],
                      tensor: mi.TensorXf | mi.TensorXu | mi.TensorXb,
                      depth: mi.UInt,
                      active: mi.Bool) -> mi.Float | mi.UInt | mi.Bool:
        r"""
        Gathers data from ``tensor`` for the specified ``depth``

        ``tensor`` is assumed to have shape `[buffer_size, depth_dim_size]`.

        :param dtype: Desired output Mitsuba type
        :param tensor: Tensor to gather from
        :param depth: Depths
        :param active: Flags specifying active components

        :return: Gathered values
        """

        valid = active & (depth > 0)

        ind = dr.arange(mi.UInt, self.buffer_size) * self.depth_dim_size \
              + depth - 1
        # There was a bug in the OptiX compiler stack which led to illegal
        # memory accesses. Although it was fixed in driver 580+, we keep this
        # workaround for backward compatibility with older drivers.
        assert dr.width(tensor.array) == self._tensor_width_lit
        ind &= (ind < self._tensor_width)

        return dr.gather(dtype, tensor.array, ind, valid)



class PathsBuffer(PathsBufferBase):
    r"""
    Class used to store the paths during their computation

    The output of a path solver is an instance of this class from which an
    instance of :class:`~sionna.rt.Paths` is built.

    :param buffer_size: Size of the buffer
    :param max_depth: Maximum depth
    :param diffraction: Whether diffraction is enabled
    """

    def __init__(self, buffer_size: int, max_depth: int, diffraction: bool):
        super().__init__(buffer_size, max_depth)

        self._diffraction = diffraction

        # Effective number of paths. This counter should be used to count the
        # number of paths effectively found by a solver.
        # Note that the buffer can be shrunk to this value using self.shrink()
        self._paths_counter = mi.UInt(0)

        # Set to True for if the path is valid
        self._valid = dr.full(dr.mask_t(mi.Float), False, buffer_size)

        # Index of the source from which the path originates
        self._src_indices = dr.zeros(mi.UInt, buffer_size)

        # Index of the target to which the path connects to
        self._tgt_indices = dr.zeros(mi.UInt, buffer_size)

        # Directions of departure and arrival as unit vectors.
        # As for the path vertices, the components are stored in separate
        # arrays, as Dr.Jit requires scatter targets to be flat arrays.
        self._k_tx_x = dr.zeros(mi.Float, buffer_size)
        self._k_tx_y = dr.zeros(mi.Float, buffer_size)
        self._k_tx_z = dr.zeros(mi.Float, buffer_size)
        #
        self._k_rx_x = dr.zeros(mi.Float, buffer_size)
        self._k_rx_y = dr.zeros(mi.Float, buffer_size)
        self._k_rx_z = dr.zeros(mi.Float, buffer_size)

        # Type of interaction (specular reflection, diffuse reflection, etc)
        self._interaction_types = dr.full(mi.TensorXu, InteractionType.NONE,
                                          [buffer_size, self.depth_dim_size])

        # Probabilities of the sampled interaction types
        # These are stored for field computation
        self._probs = dr.zeros(mi.TensorXf, [buffer_size, self.depth_dim_size])

        # Channel impulse response coefficients and delays are initialized to
        # `None`
        self._a = None
        self._tau = None
        # Doppler shifts of paths are initialized to `None`
        self._doppler = None

        # Diffracting edge properties
        # As diffraction is only supported for first order, we do not need to
        # store these properties for each depth.
        if diffraction:
            self._diffracting_wedges = \
                WedgeGeometry.build_with_size(buffer_size)
        else:
            self._diffracting_wedges = None

    @property
    def paths_counter(self):
        r"""Number of paths stored in the buffer

        :type: :py:class:`int`
        """
        return self._paths_counter

    def advance_paths_counter(self, add_n: mi.UInt32 | int) -> None:
        """Add the given number to the paths counter. Not atomic."""
        self._paths_counter += add_n

    @property
    def valid(self):
        r"""Flag indicating valid paths

        :type: :py:class:`mi.Bool`
        """
        return self._valid

    @valid.setter
    def valid(self, v):
        self._valid = v

    @property
    def source_indices(self):
        r"""Indices of the source from which the paths originates

        :type: :py:class:`mi.UInt`
        """
        return self._src_indices

    @property
    def target_indices(self):
        r"""Indices of the target to which the paths connect

        :type: :py:class:`mi.UInt`
        """
        return self._tgt_indices

    @property
    def k_tx(self):
        r"""Directions of departure as unit vectors

        :type: :py:class:`mi.Vector3f`
        """
        return mi.Vector3f(self._k_tx_x, self._k_tx_y, self._k_tx_z)

    @property
    def k_rx(self):
        r"""Directions of arrival as unit vectors, pointing from the target
        towards the direction from which the wave arrives, i.e., opposite to
        the direction of propagation of the incident wave

        :type: :py:class:`mi.Vector3f`
        """
        return mi.Vector3f(self._k_rx_x, self._k_rx_y, self._k_rx_z)

    @property
    def interaction_types(self):
        r"""Interactions types represented using
        :data:`~sionna.rt.constants.InteractionType`

        :type: :py:class:`mi.TensorXu [buffer_size, depth_dim_size]`
        """
        return self._interaction_types

    @property
    def diffracting_wedges(self):
        r"""Diffracting wedges

        :type: :py:class:`WedgeGeometry`
        """
        return self._diffracting_wedges

    @property
    def probs(self):
        r"""Probabilities of the sampled interaction types

        :type: :py:class:`mi.TensorXf [buffer_size, depth_dim_size]`
        """
        return self._probs

    @property
    def a(self):
        r"""Paths coefficients for every receive antenna pattern and transmit
        antenna pattern. ``a[n][m]`` stores the array of paths coefficients for
        the ``n`` th receive antenna pattern and the ``m`` th transmit antenna
        pattern

        :type: :py:class:`Tuple[mi.TensorXf, mi.TensorXf]`
        """
        return self._a

    @a.setter
    def a(self, v):
        self._a = v

    @property
    def tau(self):
        r"""Paths delays [s]

        :type: :py:class:`mi.Float`:
        """
        return self._tau

    @tau.setter
    def tau(self, v):
        self._tau = v

    @property
    def doppler(self):
        r""" Paths Doppler shifts [Hz]

        :type: :py:class:`mi.Float`
        """
        return self._doppler

    @doppler.setter
    def doppler(self, v):
        self._doppler = v

    def schedule(self) -> None:
        arrays = [
            self._max_depth,
            self._buffer_size,
            self._depth_dim_size,
            self._paths_counter,
            self._valid,
            self._src_indices,
            self._tgt_indices,
            self._k_tx_x,
            self._k_tx_y,
            self._k_tx_z,
            self._k_rx_x,
            self._k_rx_y,
            self._k_rx_z,
            self._interaction_types,
            self._vertices_x,
            self._vertices_y,
            self._vertices_z,
            self._shapes,
            self._primitives,
            self._probs,
            self._a,
            self._tau,
            self._doppler,
        ]
        if self._diffraction:
            arrays.append(self._diffracting_wedges)
        dr.schedule(*arrays)

    @dr.syntax
    def add_paths(self,
                  depth: mi.UInt,
                  indices: mi.UInt,
                  sample_data: int,
                  valid: mi.Bool,
                  tgt_index: mi.UInt,
                  k_tx: mi.Vector3f,
                  k_rx: mi.Vector3f,
                  active: mi.Bool) -> None:
        # pylint: disable=line-too-long
        r"""
        Adds paths to the buffer

        :param depth: Depths of the paths to add
        :param indices: Indices where to add paths in the buffer
        :param sample_data: Paths data to store
        :param valid: Flags indicating if the paths are valid, i.e., if their compute are finalized
        :param tgt_index: Targets to which the paths connect
        :param k_tx: Directions of departure of the paths as unit vectors
        :param k_rx: Directions of arrival of the paths as unit vectors,
            pointing opposite to the direction of propagation
        :param active: Flags specifying active paths. Inactive paths are not added.
        """

        depth_dim_size = self._depth_dim_size

        # Update the valid flag
        dr.scatter(self._valid, valid, indices, active=active)
        # Update the source index
        dr.scatter(self._src_indices, sample_data.src_indices, indices, active)
        # Update the target index
        dr.scatter(self._tgt_indices, tgt_index, indices, active)
        # Update directions of departure and arrival
        self._scatter_k(self._k_tx_x, self._k_tx_y, self._k_tx_z, k_tx,
                        indices, active)
        self._scatter_k(self._k_rx_x, self._k_rx_y, self._k_rx_z, k_rx,
                        indices, active)
        # Diffracting wedge
        if self._diffraction and (sample_data.diffracting_wedges is not None):
            dr.scatter(self._diffracting_wedges,
                       sample_data.diffracting_wedges,
                       indices, active)

        # Only add additional path data if `max_depth > 0`
        d = dr.ones(mi.UInt, dr.width(active))
        while d <= depth:
            interaction_types, shapes, primitives, vertices, probs \
                = sample_data.get(d, active=active)

            # Indices for updating the interaction type, shape, and primitive
            # arrays
            indices_t = indices*depth_dim_size + d - 1

            # Update interaction types
            dr.scatter(self._interaction_types.array, interaction_types,
                        indices_t, active)
            # Update shapes
            dr.scatter(self._shapes.array, shapes, indices_t, active)
            # Update primitives
            dr.scatter(self._primitives.array, primitives, indices_t, active)
            # Update vertices
            dr.scatter(self._vertices_x.array, vertices.x, indices_t,active)
            dr.scatter(self._vertices_y.array, vertices.y, indices_t,active)
            dr.scatter(self._vertices_z.array, vertices.z, indices_t,active)
            # Update probabilities
            dr.scatter(self._probs.array, probs, indices_t, active)

            d += 1

    def shrink(self) -> None:
        r"""
        Shrinks the buffer size to :attr:`~sionna.rt.PathsBuffer.paths_counter`

        Only the first :attr:`~sionna.rt.PathsBuffer.paths_counter` items are
        kept.
        """

        # Effective number of paths
        num_paths = dr.minimum(dr.max(self._paths_counter),
                               self._buffer_size)[0]

        depth_dim_size = self._depth_dim_size

        self._valid = dr.reshape(mi.Bool, self._valid, num_paths,
                                 shrink=True)
        self._src_indices = dr.reshape(mi.UInt, self._src_indices,
                                       num_paths, shrink=True)
        self._tgt_indices = dr.reshape(mi.UInt, self._tgt_indices,
                                       num_paths, shrink=True)
        if self._a is not None:
            a = []
            for a_ in self._a:
                a1 = []
                for a__ in a_:
                    a1.append(dr.reshape(mi.Complex2f, a__, num_paths,
                                         shrink=True))
                a.append(a1)
            self._a = a
        if self._tau is not None:
            self._tau = dr.reshape(mi.Float, self._tau, num_paths, shrink=True)
        if self._doppler is not None:
            self._doppler = dr.reshape(mi.Float, self._doppler, num_paths,
                                       shrink=True)
        self._k_tx_x = dr.reshape(mi.Float, self._k_tx_x,
                                  num_paths, shrink=True)
        self._k_tx_y = dr.reshape(mi.Float, self._k_tx_y,
                                  num_paths, shrink=True)
        self._k_tx_z = dr.reshape(mi.Float, self._k_tx_z,
                                  num_paths, shrink=True)
        self._k_rx_x = dr.reshape(mi.Float, self._k_rx_x,
                                  num_paths, shrink=True)
        self._k_rx_y = dr.reshape(mi.Float, self._k_rx_y,
                                  num_paths, shrink=True)
        self._k_rx_z = dr.reshape(mi.Float, self._k_rx_z,
                                  num_paths, shrink=True)
        self._interaction_types = mi.TensorXu(
            dr.reshape(mi.UInt, self._interaction_types.array,
                       num_paths*depth_dim_size, shrink=True),
            shape=(num_paths,depth_dim_size)
        )
        self._vertices_x = mi.TensorXf(
            dr.reshape(mi.Float, self._vertices_x.array,
                       num_paths*depth_dim_size, shrink=True),
            shape=(num_paths, depth_dim_size)
        )
        self._vertices_y = mi.TensorXf(
            dr.reshape(mi.Float, self._vertices_y.array,
                       num_paths*depth_dim_size, shrink=True),
            shape=(num_paths, depth_dim_size)
        )
        self._vertices_z = mi.TensorXf(
            dr.reshape(mi.Float, self._vertices_z.array,
                       num_paths*depth_dim_size, shrink=True),
            shape=(num_paths, depth_dim_size)
        )
        self._shapes = mi.TensorXu(
            dr.reshape(mi.UInt, self._shapes.array,
                       num_paths*depth_dim_size, shrink=True),
            shape=(num_paths, depth_dim_size)
        )
        self._primitives = mi.TensorXu(
            dr.reshape(mi.UInt, self._primitives.array,
                       num_paths*depth_dim_size, shrink=True),
            shape=(num_paths, depth_dim_size)
        )
        if self._diffraction:
            self._diffracting_wedges = WedgeGeometry(
            *[dr.reshape(type(getattr(self._diffracting_wedges, field.name)),
                          getattr(self._diffracting_wedges, field.name),
                          num_paths, shrink=True)
                for field in dataclasses.fields(self._diffracting_wedges)])
        self._probs = mi.TensorXf(
            dr.reshape(mi.Float, self._probs.array,
                       num_paths*depth_dim_size, shrink=True),
            shape=(num_paths, depth_dim_size)
        )

        self._buffer_size = num_paths
        self._paths_counter = num_paths
        self._tensor_width_lit = dr.width(self._shapes.array)
        self._tensor_width = dr.opaque(mi.UInt32, self._tensor_width_lit)

    def discard_duplicates(self) -> None:
        r"""
        Marks copies of the same path as invalid

        Two paths are copies if they have the same source, target, and
        sequence of interactions: at every depth the same interaction type and
        shape, plus the same primitive at specular reflections and the same
        edge at diffractions. Of each set of copies, the path with the lowest
        buffer index is kept. Paths with a diffuse reflection are never
        marked. :meth:`~sionna.rt.PathsBuffer.discard_invalid` drops the
        marked paths.
        """
        num_paths = self._buffer_size
        depth_dim_size = self._depth_dim_size
        if num_paths == 0:
            return

        # Evaluate the buffer, so that computed quantities are not re-evaluated
        self.schedule()
        dr.eval()
        interaction_types = self._interaction_types.array
        interaction_shapes = self._shapes.array
        interaction_primitives = self._primitives.array

        # Paths to check: valid ones without a diffuse reflection
        all_path_indices = dr.arange(mi.UInt, num_paths)
        has_diffuse = dr.zeros(mi.Bool, num_paths)
        for depth in range(depth_dim_size):
            interaction_type = dr.gather(
                mi.UInt, interaction_types,
                all_path_indices*depth_dim_size + depth)
            has_diffuse |= interaction_type == InteractionType.DIFFUSE
        checked_indices = dr.compress(self._valid & ~has_diffuse)
        num_checked_paths = dr.width(checked_indices)
        if num_checked_paths < 2:
            return

        def build_path_keys(path_indices):
            """Key columns of the paths at `path_indices`, left unevaluated"""
            key_columns = [dr.gather(mi.UInt, self._src_indices, path_indices),
                           dr.gather(mi.UInt, self._tgt_indices, path_indices)]
            is_diffracting = dr.zeros(mi.Bool, dr.width(path_indices))
            for depth in range(depth_dim_size):
                flat_indices = path_indices*depth_dim_size + depth
                interaction_type = dr.gather(mi.UInt, interaction_types,
                                             flat_indices)
                shape = dr.gather(mi.UInt, interaction_shapes, flat_indices)
                primitive = dr.gather(mi.UInt, interaction_primitives,
                                      flat_indices)
                is_diffracting |= \
                    interaction_type == InteractionType.DIFFRACTION
                key_columns += [interaction_type, shape,
                                dr.select(interaction_type
                                          == InteractionType.SPECULAR,
                                          primitive, INVALID_PRIMITIVE)]
            if not self._diffraction:
                return key_columns

            # The diffracting edge (at most one per path): its two faces, and
            # its two vertices, as the faces of a two-sided panel share all
            # three edges
            wedges = dr.gather(WedgeGeometry, self._diffracting_wedges,
                               path_indices)
            lower_face = dr.minimum(wedges.prim0, wedges.primn)
            upper_face = dr.select(wedges.prim0 == wedges.primn,
                                   0x80000000 | wedges.local_edge,  # a screen
                                   dr.maximum(wedges.prim0, wedges.primn))

            # `local_edge` refers to `prim0` or, after a face swap, `primn`:
            # take the edge whose end points lie on the wedge's edge line
            def measure_distance_off_edge_line(edge_vertex_indices):
                start = wedges.shape.vertex_position(edge_vertex_indices.x,
                                                     is_diffracting)
                end = wedges.shape.vertex_position(edge_vertex_indices.y,
                                                   is_diffracting)
                return dr.norm(dr.cross(start - wedges.o, wedges.e_hat))\
                    + dr.norm(dr.cross(end - wedges.o, wedges.e_hat))
            edge_on_prim0 = wedges.shape.edge_indices(wedges.prim0,
                                                      wedges.local_edge,
                                                      is_diffracting)
            edge_on_primn = wedges.shape.edge_indices(wedges.primn,
                                                      wedges.local_edge,
                                                      is_diffracting)
            edge_vertex_indices = dr.select(
                measure_distance_off_edge_line(edge_on_prim0)
                <= measure_distance_off_edge_line(edge_on_primn),
                edge_on_prim0, edge_on_primn)
            edge_columns = (lower_face, upper_face,
                            dr.minimum(edge_vertex_indices.x,
                                       edge_vertex_indices.y),
                            dr.maximum(edge_vertex_indices.x,
                                       edge_vertex_indices.y))
            return key_columns + [dr.select(is_diffracting, column,
                                            INVALID_PRIMITIVE)
                                  for column in edge_columns]

        # Sort the checked paths by key hash; the sort is stable, so the
        # copies of a path come after its lowest buffer index
        first_column, *other_columns = build_path_keys(checked_indices)
        key_hashes = hash_fnv1a(first_column)
        for key_column in other_columns:
            key_hashes = hash_fnv1a(key_column, h=key_hashes)
        sorted_order = dr.argsort(key_hashes)
        sorted_positions = dr.arange(mi.UInt, num_checked_paths)
        previous_positions = dr.gather(mi.UInt, sorted_order,
                                       dr.maximum(sorted_positions, 1) - 1)
        sorted_path_indices = dr.gather(mi.UInt, checked_indices, sorted_order)
        previous_path_indices = dr.gather(mi.UInt, checked_indices,
                                          previous_positions)

        # Compare the whole key with the previous path, so that a hash
        # collision never drops a path
        is_copy = sorted_positions > 0
        for sorted_column, previous_column in zip(
                build_path_keys(sorted_path_indices),
                build_path_keys(previous_path_indices)):
            is_copy &= sorted_column == previous_column
        dr.scatter(self._valid, False, sorted_path_indices, active=is_copy)

    def discard_invalid(self) -> None:
        r"""
        Discards invalid paths

        This function discards paths for which the entry in
        :attr:`~sionna.rt.PathsBuffer.valid` is set to `False`.
        """

        depth_dim_size = self._depth_dim_size
        self.schedule()

        # Indices of valid paths
        valid_ind = dr.compress(self._valid)

        # Number of valid paths
        num_valid_paths = dr.shape(valid_ind)[0]

        # Tensor indices for gathering the paths data
        depth_ind = dr.tile(dr.arange(mi.UInt, 0, depth_dim_size),
                            num_valid_paths)
        # Need to handle the case where no paths is found separately
        if num_valid_paths == 0:
            tensor_gind = []
        else:
            tensor_gind = dr.repeat(valid_ind, depth_dim_size)*depth_dim_size\
                            + depth_ind

        self._valid = dr.gather(mi.Bool, self._valid, valid_ind)
        self._src_indices = dr.gather(mi.UInt, self._src_indices, valid_ind)
        self._tgt_indices = dr.gather(mi.UInt, self._tgt_indices, valid_ind)
        if self._a is not None:
            a = []
            for a_ in self._a:
                a1 = []
                for a__ in a_:
                    a1.append(dr.gather(mi.Complex2f, a__, valid_ind))
                a.append(a1)
            self._a = a
        if self._tau is not None:
            self._tau = dr.gather(mi.Float, self._tau, valid_ind)
        if self._doppler is not None:
            self._doppler = dr.gather(mi.Float, self._doppler, valid_ind)
        self._k_tx_x = dr.gather(mi.Float, self._k_tx_x, valid_ind)
        self._k_tx_y = dr.gather(mi.Float, self._k_tx_y, valid_ind)
        self._k_tx_z = dr.gather(mi.Float, self._k_tx_z, valid_ind)
        self._k_rx_x = dr.gather(mi.Float, self._k_rx_x, valid_ind)
        self._k_rx_y = dr.gather(mi.Float, self._k_rx_y, valid_ind)
        self._k_rx_z = dr.gather(mi.Float, self._k_rx_z, valid_ind)
        self._interaction_types = mi.TensorXu(
            dr.gather(mi.UInt, self._interaction_types.array, tensor_gind),
            shape=(num_valid_paths, depth_dim_size)
        )
        self._vertices_x = mi.TensorXf(
            dr.gather(mi.Float, self._vertices_x.array, tensor_gind),
            shape=(num_valid_paths, depth_dim_size)
        )
        self._vertices_y = mi.TensorXf(
            dr.gather(mi.Float, self._vertices_y.array, tensor_gind),
            shape=(num_valid_paths, depth_dim_size)
        )
        self._vertices_z = mi.TensorXf(
            dr.gather(mi.Float, self._vertices_z.array, tensor_gind),
            shape=(num_valid_paths, depth_dim_size)
        )
        self._shapes = mi.TensorXu(
            dr.gather(mi.UInt, self._shapes.array, tensor_gind),
            shape=(num_valid_paths, depth_dim_size)
        )
        self._primitives = mi.TensorXu(
            dr.gather(mi.UInt, self._primitives.array, tensor_gind),
            shape=(num_valid_paths, depth_dim_size)
        )
        if self._diffraction:
            self._diffracting_wedges = dr.gather(
                WedgeGeometry,
                self._diffracting_wedges,
                valid_ind
            )
        self._probs = mi.TensorXf(
            dr.gather(mi.Float, self._probs.array, tensor_gind),
            shape=(num_valid_paths, depth_dim_size)
        )

        self._buffer_size = num_valid_paths
        self._paths_counter = num_valid_paths
        self._tensor_width_lit = dr.width(self._shapes.array)
        self._tensor_width = dr.opaque(mi.UInt32, self._tensor_width_lit)

    def gather(self, indices: mi.UInt) -> "PathsBuffer":
        # pylint: disable=protected-access
        r"""
        Extracts the paths with indices ``indices`` into a new paths buffer

        The returned buffer has a size equal to the number of items in
        ``indices``, and its :attr:`~sionna.rt.PathsBuffer.paths_counter` is set
        to this value. Paths can be selected multiple times, and the order of
        ``indices`` is preserved.

        The channel impulse response coefficients
        (:attr:`~sionna.rt.PathsBuffer.a`), delays
        (:attr:`~sionna.rt.PathsBuffer.tau`), and Doppler shifts
        (:attr:`~sionna.rt.PathsBuffer.doppler`) are only extracted if they were
        computed, i.e., if they are not `None`.

        :param indices: Indices of the paths to extract. Must be in
            `[0, buffer_size)`.

        :return: New buffer storing the selected paths
        """

        depth_dim_size = self._depth_dim_size
        self.schedule()

        # Number of extracted paths
        num_paths = dr.width(indices)

        # Tensor indices for gathering the paths data
        depth_ind = dr.tile(dr.arange(mi.UInt, 0, depth_dim_size), num_paths)
        # Need to handle the case where no path is extracted separately
        if num_paths == 0:
            tensor_gind = []
        else:
            tensor_gind = dr.repeat(indices, depth_dim_size)*depth_dim_size\
                            + depth_ind

        other = PathsBuffer(num_paths, self._max_depth, self._diffraction)

        other._valid = dr.gather(mi.Bool, self._valid, indices)
        other._src_indices = dr.gather(mi.UInt, self._src_indices, indices)
        other._tgt_indices = dr.gather(mi.UInt, self._tgt_indices, indices)
        if self._a is not None:
            a = []
            for a_ in self._a:
                a1 = []
                for a__ in a_:
                    a1.append(dr.gather(mi.Complex2f, a__, indices))
                a.append(a1)
            other._a = a
        if self._tau is not None:
            other._tau = dr.gather(mi.Float, self._tau, indices)
        if self._doppler is not None:
            other._doppler = dr.gather(mi.Float, self._doppler, indices)
        other._k_tx_x = dr.gather(mi.Float, self._k_tx_x, indices)
        other._k_tx_y = dr.gather(mi.Float, self._k_tx_y, indices)
        other._k_tx_z = dr.gather(mi.Float, self._k_tx_z, indices)
        other._k_rx_x = dr.gather(mi.Float, self._k_rx_x, indices)
        other._k_rx_y = dr.gather(mi.Float, self._k_rx_y, indices)
        other._k_rx_z = dr.gather(mi.Float, self._k_rx_z, indices)
        other._interaction_types = mi.TensorXu(
            dr.gather(mi.UInt, self._interaction_types.array, tensor_gind),
            shape=(num_paths, depth_dim_size)
        )
        other._vertices_x = mi.TensorXf(
            dr.gather(mi.Float, self._vertices_x.array, tensor_gind),
            shape=(num_paths, depth_dim_size)
        )
        other._vertices_y = mi.TensorXf(
            dr.gather(mi.Float, self._vertices_y.array, tensor_gind),
            shape=(num_paths, depth_dim_size)
        )
        other._vertices_z = mi.TensorXf(
            dr.gather(mi.Float, self._vertices_z.array, tensor_gind),
            shape=(num_paths, depth_dim_size)
        )
        other._shapes = mi.TensorXu(
            dr.gather(mi.UInt, self._shapes.array, tensor_gind),
            shape=(num_paths, depth_dim_size)
        )
        other._primitives = mi.TensorXu(
            dr.gather(mi.UInt, self._primitives.array, tensor_gind),
            shape=(num_paths, depth_dim_size)
        )
        if self._diffraction:
            other._diffracting_wedges = dr.gather(
                WedgeGeometry,
                self._diffracting_wedges,
                indices
            )
        other._probs = mi.TensorXf(
            dr.gather(mi.Float, self._probs.array, tensor_gind),
            shape=(num_paths, depth_dim_size)
        )

        other._paths_counter = mi.UInt(num_paths)

        return other

    def tile(self, count: int) -> "PathsBuffer":
        r"""
        Repeats the content of the buffer ``count`` times into a new paths
        buffer, similarly to :func:`drjit.tile`

        The paths of the returned buffer, which has a size equal to
        `buffer_size*count`, are ordered as
        :math:`(p_0, \dots, p_{N-1}, p_0, \dots, p_{N-1}, \dots)`, where
        :math:`p_i` denotes the :math:`i` th path of this buffer and :math:`N`
        its size.

        As with :meth:`~sionna.rt.PathsBuffer.gather`, the channel impulse
        response coefficients (:attr:`~sionna.rt.PathsBuffer.a`), delays
        (:attr:`~sionna.rt.PathsBuffer.tau`), and Doppler shifts
        (:attr:`~sionna.rt.PathsBuffer.doppler`) are only copied if they were
        computed, i.e., if they are not `None`.

        :param count: Number of repetitions

        :return: New buffer storing the tiled paths
        """

        indices = dr.tile(dr.arange(mi.UInt, self._buffer_size), count)
        return self.gather(indices)

    def repeat(self, count: int) -> "PathsBuffer":
        r"""
        Repeats every path of the buffer ``count`` times into a new paths
        buffer, similarly to :func:`drjit.repeat`

        The paths of the returned buffer, which has a size equal to
        `buffer_size*count`, are ordered as
        :math:`(p_0, \dots, p_0, p_1, \dots, p_1, \dots)`, where :math:`p_i`
        denotes the :math:`i` th path of this buffer.

        As with :meth:`~sionna.rt.PathsBuffer.gather`, the channel impulse
        response coefficients (:attr:`~sionna.rt.PathsBuffer.a`), delays
        (:attr:`~sionna.rt.PathsBuffer.tau`), and Doppler shifts
        (:attr:`~sionna.rt.PathsBuffer.doppler`) are only copied if they were
        computed, i.e., if they are not `None`.

        :param count: Number of repetitions

        :return: New buffer storing the repeated paths
        """

        indices = dr.repeat(dr.arange(mi.UInt, self._buffer_size), count)
        return self.gather(indices)

    def reverse(self) -> "PathsBuffer":
        # pylint: disable=protected-access, line-too-long
        r"""
        Reverses the direction of the paths

        A path connecting a source :math:`a` to a target :math:`b` through the
        vertices :math:`v_1, \dots, v_D`, i.e.,
        :math:`a \rightarrow v_1 \rightarrow \dots \rightarrow v_D \rightarrow b`,
        becomes
        :math:`b \rightarrow v_D \rightarrow \dots \rightarrow v_1 \rightarrow a`.

        To that end, the indices of the sources and targets are swapped, as
        well as the directions of departure and arrival, which both point
        from the corresponding radio device towards the adjacent path vertex.
        The data stored for every interaction, i.e., the vertices, interaction
        types, shapes, primitives, and probabilities, is reversed along the
        depth dimension.

        The channel coefficients (:attr:`~sionna.rt.PathsBuffer.a`), delays
        (:attr:`~sionna.rt.PathsBuffer.tau`), and Doppler shifts
        (:attr:`~sionna.rt.PathsBuffer.doppler`) are left unchanged. The
        diffracting wedges (:attr:`~sionna.rt.PathsBuffer.diffracting_wedges`)
        are left unchanged as well, as they describe the same wedges. Note
        however that the faces of these wedges are ordered according to the
        original direction of propagation, i.e., the 0-face is the one
        illuminated by the original source.

        This buffer is left unchanged.

        :return: New buffer storing the reversed paths
        """

        other = self.gather(dr.arange(mi.UInt, self._buffer_size))
        other._paths_counter = mi.UInt(self._paths_counter)

        if self._buffer_size == 0:
            return other

        depth_dim_size = self._depth_dim_size

        # Swap the sources and the targets
        other._src_indices, other._tgt_indices = self._tgt_indices,\
                                                 self._src_indices

        # Swap the directions of departure and arrival
        other._k_tx_x, other._k_rx_x = self._k_rx_x, self._k_tx_x
        other._k_tx_y, other._k_rx_y = self._k_rx_y, self._k_tx_y
        other._k_tx_z, other._k_rx_z = self._k_rx_z, self._k_tx_z

        # Items of the tensors storing the interactions data that correspond to
        # an interaction. The interactions of a path are stored contiguously
        # starting from a depth of one, and the remaining entries are set to
        # `InteractionType.NONE`.
        is_interaction = self._interaction_types.array != InteractionType.NONE

        # Number of interactions of every path
        num_interactions = self.num_interactions()

        # For every item of the tensors storing the interactions data, index of
        # the item it should read from, i.e., of the symmetric item with respect
        # to the middle of the path.
        # Items that do not correspond to an interaction read themselves, and
        # are therefore left unchanged.
        ind = dr.arange(mi.UInt, self._buffer_size*depth_dim_size)
        path_ind = ind // depth_dim_size
        depth_ind = ind - path_ind*depth_dim_size
        path_num_interactions = dr.gather(mi.UInt, num_interactions, path_ind)
        rev_ind = path_ind*depth_dim_size \
                  + dr.select(is_interaction,
                              path_num_interactions - depth_ind - 1, depth_ind)

        other._interaction_types = self._reverse_depth(
            mi.UInt, self._interaction_types, rev_ind)
        other._vertices_x = self._reverse_depth(mi.Float, self._vertices_x,
                                                rev_ind)
        other._vertices_y = self._reverse_depth(mi.Float, self._vertices_y,
                                                rev_ind)
        other._vertices_z = self._reverse_depth(mi.Float, self._vertices_z,
                                                rev_ind)
        other._shapes = self._reverse_depth(mi.UInt, self._shapes, rev_ind)
        other._primitives = self._reverse_depth(mi.UInt, self._primitives,
                                                rev_ind)
        other._probs = self._reverse_depth(mi.Float, self._probs, rev_ind)

        return other

    def chain(self,
              others: List["PathsBuffer"],
              max_depth: int | None = None) -> "PathsBuffer":
        # pylint: disable=protected-access, line-too-long
        r"""
        Chains the paths of this buffer with the paths of ``others`` into a new
        paths buffer

        The buffers are chained in the order in which they are given, i.e.,
        this buffer followed by ``others[0]``, ``others[1]``, and so on. Every
        buffer must store the same number of paths, and the :math:`i` th path of
        the returned buffer is the concatenation of the :math:`i` th path of
        every buffer of the chain. Its
        :attr:`~sionna.rt.PathsBuffer.max_depth` defaults to the sum of the
        maximum depths of the chained buffers, which always fits the chained
        paths.

        The chained paths originate from the sources of this buffer and connect
        to the targets of the last buffer of the chain. Accordingly, the
        directions of departure are read from this buffer and the directions of
        arrival from the last one. A chained path is flagged as valid only if
        the corresponding path of every buffer of the chain is valid.

        The data stored for every interaction, i.e., the vertices, interaction
        types, shapes, primitives, and probabilities, is concatenated along the
        depth dimension. Note that the endpoints shared by two consecutive
        buffers of the chain, i.e., the target of a buffer and the source of the
        next one, are not added as interactions. If they should be recorded as
        such, they must be added to the corresponding buffer beforehand. The
        sensing targets and sensing points of
        :data:`~sionna.rt.constants.InteractionType.SENSING` interactions are
        therefore preserved, as they are stored by the shapes and primitives.

        The channel impulse response coefficients
        (:attr:`~sionna.rt.PathsBuffer.a`), delays
        (:attr:`~sionna.rt.PathsBuffer.tau`), Doppler shifts
        (:attr:`~sionna.rt.PathsBuffer.doppler`), and diffracting wedges
        (:attr:`~sionna.rt.PathsBuffer.diffracting_wedges`) of the chained
        buffers are dropped, as they describe individual segments of the chained
        paths. They must therefore be set for the returned buffer by the caller.

        This buffer and the buffers of ``others`` are left unchanged.

        :param others: Paths buffers to chain to this one, in the order in which
            they are traversed. Must all store as many paths as this buffer.
        :param max_depth: Maximum depth of the returned buffer. Only the
            interactions of the chained paths are written, so a depth as small
            as the largest number of interactions of a chained path is enough.
            The caller must guarantee this bound, as the interactions of a path
            that exceeds it would be written over the ones of the next path.
            Defaults to the sum of the maximum depths of the chained buffers.

        :return: New buffer storing the chained paths
        """

        num_paths = self._buffer_size

        buffers = [self] + list(others)
        for buffer in buffers:
            if buffer.buffer_size != num_paths:
                raise ValueError("All the chained buffers must store the same"
                                 " number of paths")

        if max_depth is None:
            max_depth = sum(buffer.max_depth for buffer in buffers)
        result = PathsBuffer(num_paths, max_depth, False)

        # The chained paths originate from the sources of this buffer and
        # connect to the targets of the last buffer of the chain
        last = buffers[-1]
        result._valid = dr.copy(self._valid)
        for buffer in others:
            result._valid &= buffer.valid
        result._src_indices = dr.copy(self._src_indices)
        result._tgt_indices = dr.copy(last.target_indices)
        result._k_tx_x = dr.copy(self._k_tx_x)
        result._k_tx_y = dr.copy(self._k_tx_y)
        result._k_tx_z = dr.copy(self._k_tx_z)
        result._k_rx_x = dr.copy(last._k_rx_x)
        result._k_rx_y = dr.copy(last._k_rx_y)
        result._k_rx_z = dr.copy(last._k_rx_z)
        result._paths_counter = mi.UInt(num_paths)

        if num_paths == 0:
            return result

        # Every buffer writes its interactions after those of the buffers that
        # precede it in the chain
        offsets = dr.zeros(mi.UInt, num_paths)
        for buffer in buffers:
            buffer._scatter_interactions(result, offsets)
            offsets += buffer.num_interactions()

        return result

    def get_interaction_type(self,
                             depth: mi.UInt,
                             active: mi.Bool) -> mi.UInt:
        r"""
        Gathers the interactions types for the specified ``depth``

        :param depth: Depths
        :param active: Flags specifying active components

        :return: Interactions types
        """

        return self._gather_depth(mi.UInt, self._interaction_types, depth,
                                  active)


    def get_prob(self, depth: mi.UInt, active: mi.Bool) -> mi.Float:
        r"""
        Gathers the probabilities of the sampled interaction types
        for the specified ``depth``

        :param depth: Depths
        :param active: Flags specifying active components
        """
        return self._gather_depth(mi.Float, self._probs, depth, active)

    def set_k_tx(self, k_tx: mi.Vector3f, active: mi.Bool) -> None:
        r"""
        Sets the directions of departure :attr:`~sionna.rt.PathsBuffer.k_tx`

        :param k_tx: Directions of departure as unit vectors
        :param active: Flags specifying active components
        """

        indices = dr.arange(mi.UInt, self.buffer_size)
        self._scatter_k(self._k_tx_x, self._k_tx_y, self._k_tx_z, k_tx,
                        indices, active)

    def set_k_rx(self, k_rx: mi.Vector3f, active: mi.Bool) -> None:
        r"""
        Sets the directions of arrival :attr:`~sionna.rt.PathsBuffer.k_rx`

        :param k_rx: Directions of arrival as unit vectors, pointing opposite
            to the direction of propagation of the incident wave
        :param active: Flags specifying active components
        """

        indices = dr.arange(mi.UInt, self.buffer_size)
        self._scatter_k(self._k_rx_x, self._k_rx_y, self._k_rx_z, k_rx,
                        indices, active)

    def set_interaction_type(self,
                             depth: mi.UInt,
                             value: mi.UInt,
                             active: mi.Bool) -> None:
        # pylint: disable=line-too-long
        r"""
        Sets the interactions types for the specified ``depth``

        :param depth: Depths
        :param value: Interactions types represented using :data:`~sionna.rt.constants.InteractionType`
        :param active: Flags specifying active components
        """

        self._scatter_depth(self._interaction_types, depth, value, active)

    def set_vertex(self,
                   depth: mi.UInt,
                   value: mi.Point3f,
                   active: mi.Bool) -> None:
        r"""
        Sets the coordinates of the vertices of the paths for the specified
        ``depth``

        :param depth: Depths
        :param value: Vertices
        :param active: Flags specifying active components
        """

        self._scatter_depth(self._vertices_x, depth, value.x, active)
        self._scatter_depth(self._vertices_y, depth, value.y, active)
        self._scatter_depth(self._vertices_z, depth, value.z, active)

    def set_shape(self,
                  depth: mi.UInt,
                  value: mi.UInt,
                  active: mi.Bool) -> None:
        r"""
        Sets the indices of the intersected shapes for the specified ``depth``

        Invalid intersections should be represented by
        :data:`~sionna.rt.constants.INVALID_SHAPE`

        :param depth: Depths
        :param value: Indices of the intersected shapes
        :param active: Flags specifying active components
        """

        self._scatter_depth(self._shapes, depth, value, active)

    def set_primitive(self,
                      depth: mi.UInt,
                      value: mi.UInt,
                      active: mi.Bool) -> None:
        r"""
        Sets the indices to the intersected primitives for the specified
        ``depth``

        :param depth: Depths
        :param value: Indices of intersected primitives
        :param active: Flags specifying active components
        """

        self._scatter_depth(self._primitives, depth, value, active)

    def set_prob(self,
                 depth: mi.UInt,
                 value: mi.Float,
                 active: mi.Bool) -> None:
        r"""
        Sets the probabilities of the sampled interaction types for
        the specified ``depth``

        :param depth: Depths
        :param value: Probabilities of the sampled interaction types
        :param active: Flags specifying active components
        """

        self._scatter_depth(self._probs, depth, value, active)

    def detach_geometry(self) -> None:
        r"""
        Detaches the arrays storing the paths geometries (i.e., vertices and
        directions of departure and arrival) from the automatic differentiation
        computational graph
        """

        self._vertices_x = dr.detach(self._vertices_x)
        self._vertices_y = dr.detach(self._vertices_y)
        self._vertices_z = dr.detach(self._vertices_z)
        self._k_tx_x = dr.detach(self._k_tx_x)
        self._k_tx_y = dr.detach(self._k_tx_y)
        self._k_tx_z = dr.detach(self._k_tx_z)
        self._k_rx_x = dr.detach(self._k_rx_x)
        self._k_rx_y = dr.detach(self._k_rx_y)
        self._k_rx_z = dr.detach(self._k_rx_z)
        if self._diffraction:
            self._diffracting_wedges = dr.detach(self._diffracting_wedges)

    def num_interactions(self) -> mi.UInt:
        r"""
        Computes the number of interactions of every path

        The interactions of a path are stored contiguously starting from a
        depth of one, and the remaining entries are set to
        :data:`~sionna.rt.constants.InteractionType.NONE`.

        :return: Number of interactions of every path
        """

        is_interaction = self._interaction_types.array != InteractionType.NONE
        return dr.block_sum(dr.select(is_interaction, mi.UInt(1), mi.UInt(0)),
                            self._depth_dim_size)

    ###############################################
    # Internal methods
    ###############################################

    def _scatter_interactions(self,
                              dst: "PathsBuffer",
                              offsets: mi.UInt) -> None:
        # pylint: disable=protected-access
        r"""
        Scatters the data stored for every interaction of the paths, i.e., the
        vertices, interaction types, shapes, primitives, and probabilities,
        into ``dst``, shifting the depth of every path by ``offsets``

        ``dst`` must store as many paths as this buffer, and its depth
        dimension must be large enough to accomodate the shifted interactions.

        :param dst: Buffer to scatter the interactions data into
        :param offsets: For every path, depth offset at which its interactions
            are written
        """

        depth_dim_size = self._depth_dim_size

        is_interaction = self._interaction_types.array != InteractionType.NONE

        ind = dr.arange(mi.UInt, self._buffer_size*depth_dim_size)
        path_ind = ind // depth_dim_size
        depth_ind = ind - path_ind*depth_dim_size
        dst_ind = path_ind*dst.depth_dim_size + depth_ind \
                  + dr.gather(mi.UInt, offsets, path_ind)

        for dst_tensor, src_tensor in (
                (dst._interaction_types, self._interaction_types),
                (dst._vertices_x, self._vertices_x),
                (dst._vertices_y, self._vertices_y),
                (dst._vertices_z, self._vertices_z),
                (dst._shapes, self._shapes),
                (dst._primitives, self._primitives),
                (dst._probs, self._probs)):
            dr.scatter(dst_tensor.array, src_tensor.array, dst_ind,
                       is_interaction)

    def _reverse_depth(self,
                       dtype: Type[Union[mi.Float, mi.UInt, mi.Bool]],
                       tensor: mi.TensorXf | mi.TensorXu | mi.TensorXb,
                       indices: mi.UInt
        ) -> mi.TensorXf | mi.TensorXu | mi.TensorXb:
        r"""
        Reorders the items of ``tensor`` according to ``indices``

        ``tensor`` is assumed to have shape `[buffer_size, depth_dim_size]`.

        :param dtype: Mitsuba type of the items of ``tensor``
        :param tensor: Tensor to reorder
        :param indices: For every item of the flattened ``tensor``, index of the
            item to read from

        :return: Reordered tensor
        """

        return type(tensor)(dr.gather(dtype, tensor.array, indices),
                            shape=(self._buffer_size, self._depth_dim_size))

    @staticmethod
    def _scatter_k(k_x: mi.Float,
                   k_y: mi.Float,
                   k_z: mi.Float,
                   value: mi.Vector3f,
                   indices: mi.UInt,
                   active: mi.Bool) -> None:
        r"""
        Scatters the components of the direction vectors ``value`` into the
        arrays ``k_x``, ``k_y``, and ``k_z``

        :param k_x: Array storing the x components
        :param k_y: Array storing the y components
        :param k_z: Array storing the z components
        :param value: Direction vectors to insert
        :param indices: Indices at which to insert the direction vectors
        :param active: Flags specifying active components
        """

        dr.scatter(k_x, value.x, indices, active)
        dr.scatter(k_y, value.y, indices, active)
        dr.scatter(k_z, value.z, indices, active)

    def _scatter_depth(self,
                       tensor: mi.TensorXf | mi.TensorXu | mi.TensorXb,
                       depth: mi.UInt,
                       value: mi.Float | mi.UInt | mi.Bool,
                       active: mi.Bool) -> None:
        r"""
        Sets the items of ``tensor`` to ``value`` for the specified ``depth``

        ``tensor`` is assumed to have shape `[buffer_size, depth_dim_size]`.

        :param tensor: Tensor to update
        :param depth: Depths
        :param value: Value to insert in ``tensor``
        :param active: Flags specifying active components
        """

        valid = active & (depth > 0)
        ind = dr.arange(mi.UInt, self.buffer_size) * self.depth_dim_size \
              + depth - 1
        dr.scatter(tensor.array, value, ind, valid)
