"""Test Routine for component"""


from unittest import TestCase
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from copy import deepcopy


from amfe.component.component_composite import ComponentComposite
from amfe.component.structural_component import StructuralComponent


class ComponentCompositeTest(TestCase):
    def setUp(self):
        class DummyConstraint:
            def __init__(self):
                self.no_of_constrained_dofs = 2
            
            def unconstrain_u(self, u_constr):
                return np.array([0, 0, 0])
            
            def constrain_matrix(self,K_unconstr):
                return K_unconstr[0:2,0:2]

        class DummyAssembly:
            def __init__(self):
                pass
            
            def assemble_k_and_f(self, u, t):
                K_unconstr=np.array([[10, -5, 0], [-5, 10, -5], [0, -5, 10]])
                f_unsonstr=np.array([2, 0, 0])
                return K_unconstr, f_unsonstr
            
        class DummyMesh:
            def __init__(self, dimension):
                self.dimension = dimension
                self.nodes = np.empty((0, dimension), dtype=float)
                self.connectivity = list()
                self.el_df = pd.DataFrame(columns=['shape', 'is_boundary', 'connectivity_idx'])
                
            @property
            def no_of_elements(self):
                return 0
            
        self.mesh = DummyMesh(2)
        self.constraints = DummyConstraint()
        self.assembly = DummyAssembly()
        
        self.TestComponent = []
        comp = StructuralComponent(self.mesh)
        comp._assembly = self.assembly
        comp._constraints = self.constraints
        self.TestComponent.append(comp)
        self.CompComposite = ComponentComposite(self.TestComponent)

    def tearDown(self):
        pass

    def test_add_component(self):
        prev_N_components = self.CompComposite.no_of_components
        
        self.TestComponent2 = deepcopy(self.TestComponent[0])
        
        self.CompComposite.add_component(self.TestComponent2)
        
        self.assertEqual(self.CompComposite.no_of_components, prev_N_components+1)
        print('No Components: ', self.CompComposite.no_of_components)
        print(len(self.CompComposite.components))
        self.assertTrue(self.TestComponent2 in self.CompComposite.components)

    def test_delete_component(self):
        self.TestComponent2 = deepcopy(self.TestComponent)
        self.CompComposite.add_component(self.TestComponent2)
        prev_N_components = self.CompComposite.no_of_components
        
        self.CompComposite.delete_component(1)
        
        self.assertEqual(self.CompComposite.no_of_components, prev_N_components-1)
        self.assertTrue(self.TestComponent2 not in self.CompComposite.components)
        
    def test_get_mat(self):
        #self.CompComposite.add_component(self.TestComponent)
        assert_array_equal(self.CompComposite.get_mat('K', None, 0, 0), self.CompComposite.components[0].K(None, 0))


class StructuralComponentTest(TestCase):
    def setUp(self):
        class DummyConstraint:
            def __init__(self):
                self.no_of_constrained_dofs = 2
            
            def unconstrain_u(self, u_constr):
                return np.array([0, 0, 0])
            
            def constrain_matrix(self, K_unconstr):
                return K_unconstr[0:2, 0:2]

        class DummyAssembly:
            def __init__(self):
                pass
            
            def assemble_k_and_f(self, u, t):
                K_unconstr=np.array([[10, -5, 0], [-5, 10, -5], [0, -5, 10]])
                f_unsonstr=np.array([2, 0, 0])
                return K_unconstr, f_unsonstr
            
        class DummyMesh:
            def __init__(self,dimension):
                self.dimension = dimension
                self.nodes = np.empty((0, dimension), dtype=float)
                self.connectivity = list()
                self.el_df = pd.DataFrame(columns=['shape', 'is_boundary', 'connectivity_idx'])

            @property
            def no_of_elements(self):
                return 0
            
        self.mesh = DummyMesh(2)
        self.constraints = DummyConstraint()
        self.assembly = DummyAssembly()
        
        self.structComp = []
        comp = StructuralComponent(self.mesh)
        comp._constraints = DummyConstraint()
        comp._assembly = DummyAssembly()
        self.structComp.append(comp)

    def tearDown(self):
        pass

    def test_initialization(self):
        pass
        
    def test_k(self):
        desiredK=np.array([[10, -5], [-5, 10]])
        assert_array_equal(self.structComp[0].K(), desiredK)
