# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from unittest import TestCase, main
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_equal, assert_array_equal, assert_allclose
from copy import deepcopy


class TestMesh(TestCase):
    def setUp(self):
        from amfe.mesh import Mesh
        self.testmesh = Mesh(dimension=2)
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float)
        connectivity = [np.array([5, 6, 3], dtype=np.int), np.array([3, 2, 5], dtype=np.int),
                        np.array([1, 2, 3, 4], dtype=np.int),
                        # boundary elements
                        np.array([4, 1], dtype=np.int), np.array([5, 6], dtype=np.int)]

        self._connectivity = connectivity

        data = {'shape': ['Tri3', 'Tri3', 'Quad4', 'straight_line', 'straight_line'],
                'is_boundary': [False, False, False, True, True],
                'connectivity': connectivity}
        indices = [1, 2, 3, 4, 5]
        el_df = pd.DataFrame(data, index=indices)

        x = nodes[:, 0]
        y = nodes[:, 1]
        nodeids = [1, 2, 3, 4, 5, 6]
        nodes_df = pd.DataFrame({'x': x, 'y': y}, index=nodeids)

        groups = {'left': {'elements': [3], 'nodes': []},
                  'right': {'elements': [1, 2], 'nodes': [2, 3, 5, 6]},
                  'left_boundary': {'elements': [4], 'nodes': []},
                  'right_boundary': {'elements': [5], 'nodes': [1, 2]}
                  }

        self.testmesh.nodes_df = nodes_df
        self.testmesh.groups = groups
        self.testmesh._el_df = el_df

        self.testmesh3d = deepcopy(self.testmesh)
        nodes3d = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                            [2.0, 0.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float)
        self.testmesh3d.dimension = 3
        x = nodes3d[:, 0]
        y = nodes3d[:, 1]
        z = nodes3d[:, 2]
        nodes_df3d = pd.DataFrame({'x': x, 'y': y, 'z': z}, index=nodeids)
        self.testmesh3d.nodes_df = nodes_df3d

    def tearDown(self):
        pass

    def test_constructor(self):
        from amfe.mesh import Mesh
        mesh = Mesh(dimension=2)
        self.assertEqual(mesh.dimension, 2)
        mesh = Mesh(dimension=3)
        self.assertEqual(mesh.dimension, 3)
        with self.assertRaises(ValueError) as err:
            mesh = Mesh(dimension=1)

    def test_no_of_properties(self):
        self.assertEqual(self.testmesh.no_of_nodes, 6)
        self.assertEqual(self.testmesh.no_of_elements, 3)
        self.assertEqual(self.testmesh.no_of_boundary_elements, 2)
        self.assertEqual(self.testmesh.dimension, 2)

    def test_nodes_voigt(self):
        desired = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0], dtype=np.float)
        assert_equal(self.testmesh.nodes_voigt, desired)

    def test_connectivity(self):
        desireds = np.array(self._connectivity)
        actuals = self.testmesh.connectivity
        for actual, desired in zip(actuals, desireds):
            assert_array_equal(actual, desired)

    def test_get_connectivity_by_elementids(self):
        desireds = [np.array([5, 6, 3], dtype=int), np.array([4, 1], dtype=int)]
        for actual, desired in zip(self.testmesh.get_connectivity_by_elementids([1, 4]), desireds):
            assert_array_equal(actual, desired)

    def test_get_elementidxs_by_group(self):
        actual = self.testmesh.get_elementidxs_by_groups(['right'])
        desired = np.array([0, 1], dtype=int)
        assert_array_equal(actual, desired)

    def test_get_elementidxs_by_elementids(self):
        actual = self.testmesh.get_elementidxs_by_elementids([4, 1])
        desired = np.array([3, 0], dtype=int)
        assert_array_equal(actual, desired)

    def test_get_elementids_by_elementidxs(self):
        actual = self.testmesh.get_elementids_by_elementidxs([3, 0])
        desired = np.array([4, 1], dtype=int)
        assert_array_equal(actual, desired)

    def test_get_elementidxs_by_groups(self):
        actual = self.testmesh.get_elementidxs_by_groups(['right', 'left_boundary'])
        desired = np.array([0, 1, 3], dtype=int)
        assert_array_equal(actual, desired)

    def test_get_elementids_by_groups(self):
        actual = self.testmesh.get_elementids_by_groups(['right', 'left_boundary'])
        desired = np.array([1, 2, 4], dtype=int)
        assert_array_equal(actual, desired)

    def test_get_nodeid_by_coordinates(self):
        # 2d case
        x, y = 2.0, 0.0
        desired = 5
        actual = self.testmesh.get_nodeid_by_coordinates(x, y)
        self.assertEqual(actual, desired)

        # 3d Case
        z = 1.0
        actual = self.testmesh3d.get_nodeid_by_coordinates(x, y, z)
        self.assertEqual(actual, desired)

        # big tolerance
        x = 500.0
        epsilon = np.inf
        actual = self.testmesh.get_nodeid_by_coordinates(x, y, epsilon=epsilon)
        self.assertEqual(actual, desired)

        actual = self.testmesh3d.get_nodeid_by_coordinates(x, y, z, epsilon=epsilon)
        self.assertEqual(actual, desired)

        # zero return
        x = 500.0
        desired = None
        actual = self.testmesh.get_nodeid_by_coordinates(x, y)
        self.assertEqual(actual, desired)

        actual = self.testmesh3d.get_nodeid_by_coordinates(x, y, z)
        self.assertEqual(actual, desired)

    def test_get_nodeids_by_x_coordinates(self):
        x = 2.0
        epsilon = 0.1
        desired = {5, 6}
        actual = set(self.testmesh.get_nodeids_by_x_coordinates(x, epsilon))
        self.assertEqual(actual, desired)

        epsilon = 1.1
        desired = {2, 3, 5, 6}
        actual = set(self.testmesh.get_nodeids_by_x_coordinates(x, epsilon))
        self.assertEqual(actual, desired)

    def test_get_nodeids_by_lesser_equal_x_coordinates(self):
        x = 1.0
        epsilon = 0.1
        desired = {1, 2, 3, 4}
        actual = set(self.testmesh.get_nodeids_by_lesser_equal_x_coordinates(x, epsilon))
        self.assertEqual(actual, desired)

        x = 0.9
        actual = set(self.testmesh.get_nodeids_by_lesser_equal_x_coordinates(x, epsilon))
        self.assertEqual(actual, desired)

        epsilon = 0.01
        desired = {1, 4}
        actual = set(self.testmesh.get_nodeids_by_lesser_equal_x_coordinates(x, epsilon))
        self.assertEqual(actual, desired)

    def test_get_nodeids_by_greater_equal_x_coordinates(self):
        x = 1.0
        epsilon = 0.1
        desired = {2, 3, 5, 6}
        actual = set(self.testmesh.get_nodeids_by_greater_equal_x_coordinates(x, epsilon))
        self.assertEqual(actual, desired)

        x = 1.1
        actual = set(self.testmesh.get_nodeids_by_greater_equal_x_coordinates(x, epsilon))
        self.assertEqual(actual, desired)

        epsilon = 0.01
        desired = {5, 6}
        actual = set(self.testmesh.get_nodeids_by_greater_equal_x_coordinates(x, epsilon))
        self.assertEqual(actual, desired)

    def test_get_nodeids_by_group(self):
        actual = self.testmesh.get_nodeids_by_groups(['left'])
        desired = np.array([1, 2, 3, 4], dtype=np.int)
        assert_equal(actual, desired)
        actual = set(self.testmesh.get_nodeids_by_groups(['right_boundary']))
        desired = set(np.array([1, 2, 5, 6]))
        assert_equal(actual, desired)
        actual = set(self.testmesh.get_nodeids_by_groups(['left_boundary']))
        desired = set(np.array([1, 4], dtype=int))
        assert_equal(actual, desired)

    def test_get_ele_shapes_by_idxs(self):
        actual = self.testmesh.get_ele_shapes_by_elementidxs([1, 4, 2])
        desired = np.array(['Tri3', 'straight_line', 'Quad4'], dtype=object)
        assert_equal(actual, desired)

    def test_get_ele_shapes_by_ids(self):
        actual = self.testmesh.get_ele_shapes_by_ids([2, 5, 3])
        desired = np.array(['Tri3', 'straight_line', 'Quad4'], dtype=object)
        assert_equal(actual, desired)

    def test_get_nodeidxs_by_all(self):
        actual = self.testmesh.get_nodeidxs_by_all()
        desired = np.array([0, 1, 2, 3, 4, 5], dtype=np.int)
        assert_equal(actual, desired)
        
    def test_get_nodeidxs_by_nodeids(self):
        actual = self.testmesh.get_nodeidxs_by_nodeids(np.array([1, 2, 5, 6]))
        desired = np.array([0, 1, 4, 5])
        assert_equal(actual, desired)

    def test_get_nodeids_by_nodeidxs(self):
        actual = self.testmesh.get_nodeids_by_nodeidxs([3, 5, 2])
        desired = [4, 6, 3]
        assert_equal(actual, desired)
        actual = self.testmesh.get_nodeids_by_nodeidxs([3])
        desired = [4]
        assert_equal(actual, desired)

    def test_get_nodeids_by_elementids(self):
        actual = self.testmesh.get_nodeids_by_elementids(np.array([2, 3], dtype=int))
        desired = np.array([1, 2, 3, 4, 5], dtype=int)
        assert_array_equal(actual, desired)

        # test zero list:
        actual = self.testmesh.get_nodeids_by_elementids(np.array([], dtype=int))
        desired = np.array([], dtype=int)
        assert_array_equal(actual, desired)

    def test_insert_tag(self):
        current_col_num = len(self.testmesh.el_df.columns)
        tag_to_add = 'partition_id'
        self.testmesh.insert_tag(tag_to_add)
        new_col_num = len(self.testmesh.el_df.columns)
        assert_equal(current_col_num + 1, new_col_num)
        self.assertTrue(tag_to_add in self.testmesh.el_df.columns)

    def test_remove_tag(self):
        tag_name = self.testmesh.el_df.columns[1]
        current_col_num = len(self.testmesh.el_df.columns)
        self.testmesh.remove_tag(tag_name)        
        new_col_num = len(self.testmesh.el_df.columns)
        assert_equal(current_col_num, new_col_num + 1)
        self.assertFalse(tag_name in self.testmesh.el_df.columns)
    
    def test_change_tag_values_by_dict(self):        
        desired_list_1 = [4,5]
        desired_list_2 = [1,2,3]
        tag_value_dict = {}
        tag_value_dict['False'] = desired_list_1
        tag_value_dict['True'] = desired_list_2
        self.testmesh.change_tag_values_by_dict('is_boundary',tag_value_dict)
        actual_list_1 = self.testmesh.el_df[self.testmesh.el_df['is_boundary'] == 'False'].index.tolist()
        actual_list_2 = self.testmesh.el_df[self.testmesh.el_df['is_boundary'] == 'True'].index.tolist()
        assert_equal(actual_list_1, desired_list_1)
        assert_equal(actual_list_2, desired_list_2)

    def test_replace_tag_values(self):
        current_key = 'Tri3' 
        new_key = 'Tri6'
        tag_name = 'shape'
        desired = [1, 2]
        self.testmesh.replace_tag_values(tag_name,current_key,new_key)
        actual = self.testmesh.el_df[self.testmesh.el_df[tag_name] == new_key].index.tolist()
        assert_equal(desired, actual)
        
    def test_get_elementids_by_tags(self):
        desired = np.array([1, 2], dtype=int)
        actual = self.testmesh.get_elementids_by_tags(['shape', 'is_boundary'], ['Tri3', False])
        assert_array_equal(desired, actual)
        
    def test_get_nodeids_by_tag(self):
        desired = np.array([1, 4, 5, 6], dtype=int)
        actual = self.testmesh.get_nodeids_by_tags('shape', 'straight_line')
        assert_array_equal(desired, actual)

    def test_get_elementidxs_by_tags(self):
        desired = np.array([0, 1], dtype=int)
        actual = self.testmesh.get_elementidxs_by_tags('shape','Tri3')
        assert_equal(desired, actual)

    def test_get_iconnectivity_by_elementids(self):
        desired = np.array([np.array([0, 1, 2, 3], dtype=int), np.array([4, 5, 2], dtype=int)])
        actual = self.testmesh.get_iconnectivity_by_elementids(np.array([3, 1], dtype=int))
        for actual_arr, desired_arr in zip(actual, desired):
            assert_array_equal(desired_arr, actual_arr)

        # Ask a second time to test lazy evaluation:
        actual = self.testmesh.get_iconnectivity_by_elementids(np.array([3, 1], dtype=int))
        for actual_arr, desired_arr in zip(actual, desired):
            assert_array_equal(desired_arr, actual_arr)
            
    def test_get_groups_by_elementids(self):
        groups_actual = self.testmesh.get_groups_by_elementids([1, 2, 4])
        groups_desired = ['right', 'left_boundary']
        
        assert_equal(groups_actual, groups_desired)
        
        self.testmesh.groups = {'left': {'elements': [3], 'nodes': []},
                  'right': {'elements': [1, 2], 'nodes': [2, 3, 5, 6]},
                  'left_boundary': {'elements': [4], 'nodes': []},
                  'right_boundary': {'elements': [5, 2], 'nodes': [1, 2]}
                  }
        
        groups_actual = self.testmesh.get_groups_by_elementids([1, 2, 4])
        groups_desired = ['right', 'left_boundary', 'right_boundary']
        
        assert_equal(groups_actual, groups_desired)
        
    def test_get_groups_by_nodeids(self):
        groups_actual = self.testmesh.get_groups_by_nodeids([1, 2, 6])
        groups_desired = ['right', 'right_boundary']
        
        assert_equal(groups_actual, groups_desired)
        
    def test_get_groups_dict_by_elementids(self):
        groups_actual = self.testmesh.get_groups_dict_by_elementids([1, 2, 4])
        groups_desired = {'right': {'elements': [1, 2]}, 'left_boundary': {'elements': [4]}}
        
        assert_equal(groups_actual, groups_desired)
        
        self.testmesh.groups = {'left': {'elements': [3], 'nodes': []},
                  'right': {'elements': [1, 2], 'nodes': [2, 3, 5, 6]},
                  'left_boundary': {'elements': [4], 'nodes': []},
                  'right_boundary': {'elements': [5, 2], 'nodes': [1, 2]}
                  }
        
        groups_actual = self.testmesh.get_groups_dict_by_elementids([1, 2, 4])
        groups_desired = {'right': {'elements': [1, 2]}, 'left_boundary': {'elements': [4]}, 'right_boundary': {'elements': [2]}}
        
        assert_equal(groups_actual, groups_desired)
        
    def test_get_groups_dict_by_nodeids(self):
        groups_actual = self.testmesh.get_groups_dict_by_nodeids([1, 2, 6])
        groups_desired = {'right': {'nodes': [2, 6]},
                  'right_boundary': {'nodes': [1, 2]}
                  }
        
        assert_equal(groups_actual, groups_desired)

        
    def test_merge_into_groups(self):
        add_groups = {'left': {'elements': [3], 'nodes': [1, 4]},
                      'right': {'elements': [1, 2]},
                      'left_boundary': {'elements': [4], 'nodes': []},
                      'right_boundary': {'elements': [5, 2], 'nodes': []},
                      'new_node_group': {'nodes': [1, 2, 3, 4]}
                      }
        
        self.testmesh.merge_into_groups(add_groups)
        groups_desired = {'left': {'elements': [3], 'nodes': [1, 4]},
                          'right': {'elements': [1, 2], 'nodes': [2, 3, 5, 6]},
                          'left_boundary': {'elements': [4], 'nodes': []},
                          'right_boundary': {'elements': [5, 2], 'nodes': [1, 2]},
                          'new_node_group': {'elements': [], 'nodes': [1, 2, 3, 4]}
                          }
        
        for group_desired in groups_desired:
            for secondary_group in ['elements', 'nodes']:
                assert_equal(set(self.testmesh.groups[group_desired][secondary_group]),
                             set(groups_desired[group_desired][secondary_group]))
        self.assertEqual(len(self.testmesh.groups), len(groups_desired))
            
    def test_add_node(self):
        nodes_df_old = deepcopy(self.testmesh.nodes_df)
        nodes_df_desired = deepcopy(nodes_df_old)
        nodes_df_desired.loc[7] = {'x': 1.0, 'y': 0.5}
        
        self.testmesh.add_node(pd.Series({'x': 1.0, 'y': 0.5}))
        
        assert_frame_equal(self.testmesh.nodes_df, nodes_df_desired)
        assert_equal(self.testmesh.no_of_nodes, 7)
        
        self.testmesh.nodes_df = nodes_df_old
        nodes_df_desired = pd.DataFrame({'x': [0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0], 'y': [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5]}, index=[1, 2, 3, 4, 5, 6, 7])
        
        self.testmesh.add_node(np.array([1.0, 0.5, 4.0]), 3)
        
        assert_frame_equal(self.testmesh.nodes_df, nodes_df_desired)
        assert_equal(self.testmesh.no_of_nodes, 7)
        
    def test_copy_node_by_id(self):
        nodes_df_old = deepcopy(self.testmesh.nodes_df)
        nodes_df_desired = pd.DataFrame({'x': [0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0], 'y': [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]}, index=[1, 2, 3, 4, 5, 6, 7])
        
        self.testmesh.copy_node_by_id(3)

        assert_frame_equal(self.testmesh.nodes_df, nodes_df_desired)

    def test_iconnectivity(self):
        actual = self.testmesh.iconnectivity
        desired = [np.array([4, 5, 2], dtype=np.int), np.array([2, 1, 4], dtype=np.int),
                        np.array([0, 1, 2, 3], dtype=np.int),
                        # boundary elements
                        np.array([3, 0], dtype=np.int), np.array([4, 5], dtype=np.int)]
        for actual_arr, desired_arr in zip(actual, desired):
            assert_array_equal(desired_arr, actual_arr)
            
class TestPartitionedMesh(TestCase):
    def setUp(self):
        from amfe.mesh import Mesh
        self.testmesh = Mesh(dimension=2)
        '''
        Testmesh:                Partition:
        
                                    9---10--11   11--12
        9---10--11--12              |  *3*  |    |*4*|
        |   |   |   |               |       |    |   |
        |   |   |   |               4---3---6    6---7
        4---3---6---7            
        |   |\  |  /|               4---3---6    6---7
        |   |  \| / |               |  *1*  |    |*2*|
        1---2---5---8               |       |    |   |
                                    1---2---5    5---8
        '''
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0], [3.0, 1.0], [3.0, 0.0], [0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0]], dtype=np.float)
        connectivity = [np.array([5, 6, 3], dtype=np.int), np.array([3, 2, 5], dtype=np.int),
                        np.array([1, 2, 3, 4], dtype=np.int), np.array([5, 7, 8], dtype=np.int), np.array([6, 7, 5], dtype=np.int),
                        np.array([3, 4, 9, 10], dtype=np.int), np.array([6, 7, 11, 12], dtype=np.int), np.array([3, 6, 10, 11], dtype=np.int),
                        # boundary elements
                        np.array([4, 1], dtype=np.int), np.array([4, 9], dtype=np.int), np.array([7, 8], dtype=np.int), np.array([7, 12], dtype=np.int)]

        self._connectivity = connectivity

        data = {'shape': ['Tri3', 'Tri3', 'Quad4', 'Tri3', 'Tri3', 'Quad4', 'Quad4', 'Quad4', 'straight_line', 'straight_line', 'straight_line', 'straight_line'],
                'is_boundary': [False, False, False, False, False, False, False, False, True, True, True, True],
                'connectivity': connectivity,
                'no_of_mesh_partitions': [4, 3, 2, 3, 4, 2, 4, 4, 2, 2, 2, 2],
                'partition_id': [1, 1, 1, 2, 2, 3, 4, 3, 1, 3, 2, 4],
                'partitions_neighbors': [(2, 3, 4), (2, 3), 3, (1, 4), (1, 3, 4), 1, (1, 2, 3), (1, 2, 4), 3, 1, 4, 2]
                }
        indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        el_df = pd.DataFrame(data, index=indices)

        x = nodes[:, 0]
        y = nodes[:, 1]
        nodeids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        nodes_df = pd.DataFrame({'x': x, 'y': y}, index=nodeids)

        groups = {'left': {'elements': [3], 'nodes': []},
                  'right': {'elements': [1, 2], 'nodes': [2, 3, 5, 6]},
                  'left_boundary': {'elements': [9, 10], 'nodes': [1, 4]},
                  'right_boundary': {'elements': [11, 12], 'nodes': [7, 8]}
                  }

        self.testmesh.nodes_df = nodes_df
        self.testmesh.groups = groups
        self.testmesh._el_df = el_df
        
    def test_get_nodes_and_elements_by_partition_id(self):
        nodes_coord_desired = np.array([[2.0, 0.0], [2.0, 1.0], [3.0, 1.0], [3.0, 0.0]], dtype=np.float)
        x = nodes_coord_desired[:, 0]
        y = nodes_coord_desired[:, 1]
        nodeids = [5, 6, 7, 8]
        nodes_desired = pd.DataFrame({'x': x, 'y': y}, index=nodeids)
        
        connectivity_desired = [np.array([5, 7, 8], dtype=np.int), np.array([6, 7, 5], dtype=np.int), np.array([7, 8], dtype=np.int)]
        data = {'shape': ['Tri3', 'Tri3', 'straight_line'],
                'is_boundary': [False, False, True],
                'connectivity': connectivity_desired,
                'no_of_mesh_partitions': [3, 4, 2],
                'partition_id': [2, 2, 2],
                'partitions_neighbors': [(1, 4), (1, 3, 4), 4]
                }
        indices = [4, 5, 11]
        elements_desired = pd.DataFrame(data, index=indices)
        
        nodes, elements = self.testmesh.get_nodes_and_elements_by_partition_id(2)
        
        assert_frame_equal(nodes, nodes_desired)
        assert_frame_equal(elements, elements_desired)
        
    def test_get_neighbor_partitions(self):
        neighbors_desired = [2, 3]
        
        neighbors_actual = self.testmesh.get_neighbor_partitions(2)
        
        assert_array_equal(neighbors_actual, neighbors_desired)
        
    def test_update_connectivity_with_new_node(self):
        connectivity_desired = np.array([np.array([5, 6, 3], dtype=np.int), np.array([3, 2, 5], dtype=np.int),
                        np.array([1, 2, 3, 4], dtype=np.int), np.array([5, 13, 8], dtype=np.int), np.array([6, 7, 5], dtype=np.int),
                        np.array([3, 4, 9, 10], dtype=np.int), np.array([6, 13, 11, 12], dtype=np.int), np.array([3, 6, 10, 11], dtype=np.int),
                        # boundary elements
                        np.array([4, 1], dtype=np.int), np.array([4, 9], dtype=np.int), np.array([7, 8], dtype=np.int), np.array([7, 12], dtype=np.int)])
        
        self.testmesh.update_connectivity_with_new_node(7, 13, [4, 7])
        connectivity_actual = self.testmesh.get_connectivity_by_elementids([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        for i in range(connectivity_desired.shape[0]):
            assert_array_equal(connectivity_actual[i], connectivity_desired[i])
            
    def test_get_elementids_by_tags(self):
        desired = np.array([1, 2, 4, 5, 7, 8], dtype=int)
        actual = self.testmesh.get_elementids_by_tags('no_of_mesh_partitions', 2, True)
        assert_array_equal(desired, actual)


        


if __name__ == '__main__':
    main()
