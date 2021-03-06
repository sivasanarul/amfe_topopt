# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information.
#
"""
Example: Cantilever beam loaded at tip solved with adaptive time stepping.
"""


# load packages
import amfe
import numpy as np


# define in- and output files
input_file = amfe.amfe_dir('meshes/gmsh/chimney/Chimney200x80Hex20.msh')
output_file = amfe.amfe_dir('results/chimney/Chimney200x80Hex20_nonlinear_dynamics_generalizedalpha')


# define system
material = amfe.KirchhoffMaterial(E=2.1e11, nu=0.3, rho=7.867e3)
system = amfe.MechanicalSystem()
system.load_mesh_from_gmsh(input_file, 1, material)
system.apply_dirichlet_boundaries(2, 'xy')
ndof = system.dirichlet_class.no_of_constrained_dofs
system.apply_neumann_boundaries(key=4, val=1.0e4, direct=(1, 1, 0), time_func=lambda t: np.sin(180.32*t))
# system.apply_rayleigh_damping(1e0, 1e-5)


# vibration modes
# amfe.vibration_modes(mechanical_system=system, n=10, save=True)
# system.export_paraview(output_file + '_vibration_modes')


# define simulation parameters
options = {
    't0': 0.0,
    't_end': 10.0,
    'dt': 1.0e-4,
    'output_frequency': 1,
    'rho_inf': 0.95,
    'initial_conditions': {
        'q0': np.zeros(ndof),
        'dq0': np.zeros(ndof)},
    'relative_tolerance': 1.0E-3,
    'absolute_tolerance': 1.0E-6,
    'verbose': True,
    'max_number_of_iterations': 5,
    'convergence_abort': True,
    'write_iterations': False,
    'track_iterations': False,
    'save_solution': True}


# solve system
solver = amfe.GeneralizedAlphaNonlinearDynamicsSolver(mechanical_system=system, **options)
# solver.solve()
solver.solve_with_adaptive_time_step(dt_start=1.0e-5, dt_min=1.0e-6, dt_max=1.0e-1, change_factor_min=0.5,
                                     change_factor_max=2.0, safety_factor=0.95, failing_newton_convergence_factor=0.5,
                                     trust_value=0.01, relative_dt_tolerance=1.0e-1, max_dt_iterations=100,
                                     failing_dt_convergence_abort=False)


# write output
system.export_paraview(output_file)

end = len(system.T_output)
file = open(output_file + '.dat', 'w')
for i in range(end):
    # file.write(str(system.T_output[i]) + ' ' + str(system.u_output[i][30]) + ' ' + str(system.u_output[i][31]) \
    #            + ' ' + str(system.u_output[i][32]) + ' ' + str(system.u_output[i][36]) + ' ' \
    #            + str(system.u_output[i][37]) + ' ' + str(system.u_output[i][38]) + '\n')
    file.write(str(system.T_output[i]) + ' ' + str(system.u_output[i][30]) + ' ' + str(system.u_output[i][31]) \
               + ' ' + str(system.u_output[i][32]) + ' ' + str(system.u_output[i][36]) + ' ' \
               + str(system.u_output[i][37]) + ' ' + str(system.u_output[i][38]) + ' ' + str(solver.dt_info[i]) + '\n')
file.close()

