#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Module, that creates simple geometries and meshes them. Especially usefull for convergence studies and creation of
test-cases.
"""
from amfe.mesh import Mesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

__all__ = ['RectangleMesh2D',
           'MeshPlotter']


class RectangleMesh2D:
    def __init__(self, l, h, n_ele_l, n_ele_h, x_start=0.0, y_start=0.0):
        self.l = l
        self.h = h
        self.n_ele_l = n_ele_l
        self.n_ele_h = n_ele_h
        self.delta_h_x = self.l / self.n_ele_l
        self.delta_h_y = self.h / self.n_ele_h
        self.x_start = x_start
        self.y_start = y_start

        self._mesh = Mesh(2)

    @property
    def mesh(self):
        return self._mesh

    def generate_mesh(self):
        self._create_nodes()
        self._create_elements()

    def _create_nodes(self):
        for i_node_y in np.arange(self.n_ele_h+1):
            for i_node_x in np.arange(self.n_ele_l+1):
                self._mesh.add_node([self.x_start + i_node_x * self.delta_h_x, self.y_start + i_node_y * self.delta_h_y])

    def _create_elements(self):
        for i_ele_y in np.arange(self.n_ele_h):
            for i_ele_x in np.arange(self.n_ele_l):
                node1 = self._mesh.get_nodeid_by_coordinates(self.x_start + i_ele_x * self.delta_h_x,
                                                             self.y_start + i_ele_y * self.delta_h_y)
                node2 = self._mesh.get_nodeid_by_coordinates(self.x_start + (i_ele_x + 1) * self.delta_h_x,
                                                             self.y_start + i_ele_y * self.delta_h_y)
                node3 = self._mesh.get_nodeid_by_coordinates(self.x_start + (i_ele_x + 1) * self.delta_h_x,
                                                             self.y_start + (i_ele_y + 1) * self.delta_h_y)
                node4 = self._mesh.get_nodeid_by_coordinates(self.x_start + i_ele_x * self.delta_h_x,
                                                             self.y_start + (i_ele_y + 1) * self.delta_h_y)
                connectivity = np.array([node1, node2, node3, node4])
                self._mesh.add_element('Quad4', connectivity)

    def add_boundary(self, edge, group_name=None):
        if edge is 'left':
            nodes = self._mesh.get_nodeids_by_coordinate_axis(self.x_start, 'x', 1e-6)
        elif edge is 'right':
            nodes = self._mesh.get_nodeids_by_coordinate_axis(self.x_start+self.l, 'x', 1e-6)
        elif edge is 'bottom':
            nodes = self._mesh.get_nodeids_by_coordinate_axis(self.y_start, 'y', 1e-6)
        elif edge is 'top':
            nodes = self._mesh.get_nodeids_by_coordinate_axis(self.y_start + self.h, 'y', 1e-6)
        else:
            raise ValueError('Unknown edge-type')
        for idx in range(nodes.size-1):
            eleid = self._mesh.add_element('straight_line', np.array([nodes[idx], nodes[idx+1]]))
            if group_name is not None and len(self._mesh.get_groups_by_elementids([eleid])) == 0:
                if group_name in self._mesh.groups:
                    self._mesh.add_element_to_groups(eleid, [group_name])
                else:
                    self._mesh.create_group(group_name, (), [eleid])

    def set_checkerboard_groups(self, segment_length, segment_height, segment_groups):
        n_seg_x = int(round(self.l/segment_length))
        n_seg_y = int(round(self.h/segment_height))
        segments = dict()
        seg = 0
        for seg_y in range(n_seg_y):
            for seg_x in range(n_seg_x):
                if seg_y % 2 == 0:
                    group_name = segment_groups[seg_x % len(segment_groups)]
                else:
                    group_name = segment_groups[(seg_x+1) % len(segment_groups)]
                segments[seg] = {'nodes': {'lowerleft': np.array([seg_x * segment_length, seg_y * segment_height]),
                                           'lowerright': np.array([(seg_x + 1) * segment_length, seg_y * segment_height]),
                                           'upperleft': np.array([seg_x * segment_length, (seg_y + 1) * segment_height]),
                                           'upperright': np.array([(seg_x + 1) * segment_length, (seg_y + 1) * segment_height])},
                                 'group': group_name
                                 }
                seg += 1

        def check_node_in_segment(P1, P2, P3, P4, node):
            edge_x = P2 - P1
            edge_y = P4 - P1
            constraint_x = (np.dot(edge_x, P1) <= np.dot(edge_x, node) <= np.dot(edge_x, P2))
            constraint_y = (np.dot(edge_y, P1) <= np.dot(edge_y, node) <= np.dot(edge_y, P4))
            return constraint_x and constraint_y

        eleids = self._mesh.get_elementids_by_tags(['shape'], ['Quad4'])
        for eleid in eleids:
            nodes = self._mesh.get_nodeids_by_elementids([eleid])
            for idx, seg in segments.items():
                seg_nodes = seg['nodes']
                ele_in_segment = True
                for node in nodes:
                    node_pos = np.array([self._mesh.nodes_df.at[node,'x'], self._mesh.nodes_df.at[node,'y']])
                    if not check_node_in_segment(seg_nodes['lowerleft'], seg_nodes['lowerright'],
                                                 seg_nodes['upperright'], seg_nodes['upperleft'], node_pos):
                        ele_in_segment = False
                        break
                if ele_in_segment:
                    if seg['group'] in self._mesh.groups:
                        self._mesh.add_element_to_groups(eleid, [seg['group']])
                    else:
                        self._mesh.create_group(seg['group'], (), [eleid])
                    break

    def set_homogeneous_group(self, groupname):
        self._mesh.create_group(groupname, (), self._mesh.get_elementids_by_tags(['shape'], ['Quad4']))


class MeshPlotter:
    def __init__(self, meshes_list):
        self._meshes_list = meshes_list

    def plot_meshes(self):
        fig = plt.figure()
        for mesh in self._meshes_list:
            for idx, element in mesh._el_df.iterrows():
                nodes = element['connectivity']
                nodes = np.append(nodes, nodes[0])
                coordinates = {'x': [], 'y': []}
                for nodeid in nodes:
                    for axis in ['x', 'y']:
                        coordinates[axis].append(mesh.nodes_df.at[nodeid, axis])
                plt.plot(coordinates['x'], coordinates['y'], marker='o', color='k')
        plt.show()
        return fig

    def plot_mesh_groups(self):
        fig, ax = plt.subplots()
        x_max = 0
        y_max = 0
        groups_list = []
        for mesh in self._meshes_list:
            groups_list.extend(list(mesh.groups.keys()))
        if len(groups_list) <= 2:
            group_colors = ['b', 'r']
        elif len(groups_list) <= 7:
            group_colors = list(mcolors.BASE_COLORS.keys())
        elif len(groups_list) <= 10:
            group_colors = list(mcolors.TABLEAU_COLORS.keys())
        else:
            raise ValueError('To many groups for supported color-tables')
        color_idx = 0
        for mesh in self._meshes_list:
            for group_name, group in mesh.groups.items():
                for ele_id in group['elements']:
                    nodes = mesh._el_df.at[ele_id, 'connectivity']
                    coordinates = np.array([[]])
                    for nodeid in nodes:
                        node_pos = np.array([[]])
                        for axis in ['x', 'y']:
                            node_pos = np.append(node_pos, mesh.nodes_df.at[nodeid, axis])
                        if node_pos[0] > x_max:
                            x_max = node_pos[0] + 0.1
                        if node_pos[1] > y_max:
                            y_max = node_pos[1] + 0.1
                        if coordinates.size is 0:
                            coordinates = node_pos[np.newaxis].T
                        else:
                            coordinates = np.concatenate((coordinates, node_pos[np.newaxis].T), axis=1)
                    if len(nodes) > 2:
                        polygon = mpatches.Polygon(coordinates.T, True, edgecolor='k', facecolor=group_colors[color_idx])
                        ax.add_patch(polygon)
                    else:
                        plt.plot(coordinates[0, :], coordinates[1, :], color=group_colors[color_idx], linewidth=2)
                color_idx += 1
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        plt.show()
        return fig
