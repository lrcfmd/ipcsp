"""
This is the code to reproduce Table 1 and assess performance of a D-Wave quantum annealer for CSP.
The latter by default uses simulated annealing implementation on a classical computer provided by
D-Wave and a quantum annealer can be accessed after a registration with a single parameter change.

Some tests can be omitted by toggling 'test' value to False in the settings dictionary below.
Some parameters of the configuration spaces can be changed there as well.

Integer programming solver Gurobi is used to solve the following structures:
1. SrTiO3 for different supercells (perovskite structure)
2. Y2O3 (bixbyite structure)
3. Y2Ti2O7 (pyrochlore structure)
4. MgAl2O4 (spinel structure)
5. Ca3Al2Si3O12 (garnet structure)

Quantum experiments are presented for SrO, SrTiO3, ZnS, ZrO2.
"""

from time import time
import os
import shutil

from tabulate import tabulate
import pandas as pd

from ipcsp import root_dir
from ipcsp.integer_program import Allocate
from ipcsp.matrix_generator import Phase
from ase.calculators.gulp import GULP
import ase.io
from copy import deepcopy

'''
 The settings dictionary lists the predictions to run and parameters of the configuration spaces for
 integer programs. It is divided in two parts: to reproduce Table 1 in the paper and quantum experiments
 
 Common parameters:
   'test' -- True if the test is chosen to run
   'multiple' -- the number of repeats of the unit cell per direction, essentially, we are predicting multiple
     copies of the structure simultaneously (mostly relevant for SrTiO3)
  'group' -- space group of an allocation. Positions that are equal under symmetry, will have the same atoms
  'grid' -- the parameter g equal to the number of points per the side of a unit cell
  'top' -- the number of the lowest energy solutions that will be computed by Gurobi
    If top is 1, then only the global optimum will be considered. Note, top > 1 can occasionally produce
    solutions violating constraints. This is a bug, which should be fixed in future versions of Gurobi. 
    We simply filter out incorrect solutions for the time being.
   
 Quantum annealing specific parameters: 
   'at_dwave' -- True will connect to a D-Wave quantum annealer and use your computational budget (register first)
   'at_dwave' -- False will rely on the local simulated annealing
   'num_reads' -- the number of solutions that will be sampled using annealing
    'annealing_time' -- how long the quantum annealing will take per sample. Slower "readouts" can occasionally 
      lead to better results occasionally
    'infinity_placement' and 'infinity_orbit' are parameters gamma and mu defined in the paper to energetically 
       penalise allocations that have incorrect stoichiometry and have two atoms on top of each other
       
Note, Gurobi is called first in the quantum section as well as a shortcut to generate coefficients 
of the the integer program corresponding to the periodic lattice atom allocation problem.
It is written into model.lp. The file lp_to_bqm.py contains tools to convert this model into a QUBO problem
that can be submitted to the quantum annealer. We don't do a local minimisation step here as the structures
are relatively simple. 
'''

settings = {

    # The first part is done using Gurobi on a local machine.
    # Every line corresponds to a row in Table 1.

    # perovskite structure of SrTiO
    'SrTiO3_1': {'test': True, 'multiple': 1, 'group': 1, 'top': 1, 'grid': 4},  # group is 221, sub 195
    'SrTiO3_2': {'test': True, 'multiple': 2, 'group': 195, 'top': 1, 'grid': 8},  # group is 221, sub 195
    'SrTiO3_3': {'test': True, 'multiple': 3, 'group': 221, 'top': 1, 'grid': 6},  # group is 221, sub 195
    'SrTiO3_4': {'test': True, 'multiple': 3, 'group': 200, 'top': 1, 'grid': 6},  # group is 221, sub 195
    'SrTiO3_5': {'test': True, 'multiple': 3, 'group': 195, 'top': 1, 'grid': 6},  # group is 221, sub 195
    # bixbyite structure of Y2O3f
    'Y2O3_1': {'test': True, 'group': 206, 'top': 1, 'grid': 8},  # group is 206, sub 199, sub 198
    'Y2O3_2': {'test': True, 'group': 199, 'top': 1, 'grid': 8},  # group is 206, sub 199, sub 198
    'Y2O3_3': {'test': True, 'group': 206, 'top': 1, 'grid': 16},  # group is 206, sub 199, sub 198
    # pyrochlore structure of Y2Ti2O7
    'Y2Ti2O7_1': {'test': True, 'group': 227, 'top': 2, 'grid': 8},  # group is 227, 196, 195
    'Y2Ti2O7_2': {'test': True, 'group': 227, 'top': 1, 'grid': 16},  # group is 227, 196, 195
    # spinel structure of MgAl2O4
    'MgAl2O4_1': {'test': True, 'group': 227, 'top': 1, 'grid': 8},  # group is 227, sub 195, 196, grid is 8
    'MgAl2O4_2': {'test': True, 'group': 227, 'top': 1, 'grid': 16},  # group is 227, sub 195, 196, grid is 8
    'MgAl2O4_3': {'test': True, 'group': 196, 'top': 1, 'grid': 8},  # group is 227, sub 195, 196, grid is 8
    'MgAl2O4_4': {'test': True, 'group': 195, 'top': 10, 'grid': 8},  # group is 227, sub 195, 196, grid is 8
    # garnet structure of Ca3Al2Si3O12
    'Ca3Al2Si3O12_1': {'test': True, 'group': 230, 'top': 1, 'grid': 16},  # group is 230, sub 206, 199, grid is 8
    'Ca3Al2Si3O12_2': {'test': True, 'group': 206, 'top': 1, 'grid': 8},  # group is 230, sub 206, 199, grid is 8

    # Quantum annealer section. We use D-Wave SDK to solve the periodic lattice atom allocation problems.

    'quantum_SrO': {'test': True, 'group': 195, 'at_dwave': False, 'num_reads': 100,
                    'multiple': 1, 'infinity_placement': 100, 'infinity_orbit': 100,
                    'annealing_time': 200},  # 'group': 225; pass group 195
    'quantum_SrTiO3': {'test': True, 'group': 221, 'at_dwave': False, 'num_reads': 100,
                       'multiple': 1, 'infinity_placement': 100, 'infinity_orbit': 100,
                       'annealing_time': 200},  # group 221, passed 221
    'quantum_ZnS': {'test': True, 'group': 195, 'at_dwave': False, 'num_reads': 200,
                    'multiple': 1, 'infinity_placement': 100, 'infinity_orbit': 100,
                    'annealing_time': 200},  # 'group': 216; passed 196
    # passed 195; 168 qubits, chains of length 7
    # it has close energy minima -134 and -133
    'quantum_ZrO2': {'test': True, 'group': 198, 'at_dwave': False, 'num_reads': 300,
                     'multiple': 1, 'infinity_placement': 50, 'infinity_orbit': 50,
                     'annealing_time': 1000},  # group 225, sub 200, 198, 195
}


def process_results(lib, results, ions_count, test_name, printing=False):
    # path hack here
    os.mkdir(os.path.join("..", "results", test_name))
    calc = GULP(keywords='single', library=os.path.join(".", lib))

    # stash the allocations for future
    results_ip = deepcopy(results)

    # Compute the number of atoms
    N_atoms = 0
    for k, v in ions_count.items():
        N_atoms += v

    init = [0] * len(results)
    final = [0] * len(results)
    best_val = 0
    best_idx = 0
    # ase.io.write("best_ipcsp.vasp", results[0])
    # ase.io.write(os.path.join("..", "results", test_name, "ip_optimum.vasp"), results[0])
    print("Processing and locally optimising solutions from the integer program\n")
    for idx, cryst in enumerate(results):
        if len(cryst.arrays['positions']) == N_atoms:
            cryst.calc = calc
            init[idx] = cryst.get_potential_energy()
            # print("Initial:", init[idx])
        else:
            print("GULP received a bad solution. Gurobi's implementation of pooling occasionally provides solutions "
                  "that do not satisfy constraints. It should be corrected in future versions of the solver.")

    calc.set(keywords='opti conjugate conp diff comp c6')
    prev_energy = -1000000
    for idx, cryst in enumerate(results):
        if init[idx] < -0.00001:
            if init[idx] - prev_energy > 0.000001:
                prev_energy = init[idx]
                opt = calc.get_optimizer(cryst)
                try:
                    opt.run(fmax=0.05)
                    final[idx] = cryst.get_potential_energy()
                except ValueError:
                    print("One of the relaxations failed using initial energy instead")
                    final[idx] = init[idx]

                if final[idx] < best_val:
                    best_idx = idx
                    best_val = final[idx]
                # print("Final:", final[idx])
                # input()
            # print("Energy initial: ", cryst.get_potential_energy(), " final: ", final)

    count = 1
    with open(os.path.join("..", "results", test_name, "energies.txt"), "w+") as f:
        for i in range(len(results)):
            if final[i] != 0:
                print(f"Solution{count}: ", "Energy initial: ", init[i], " final: ", final[i])
                print(f"Solution{count}: ", "Energy initial: ", init[i], " final: ", final[i], file=f)
                # if len(results) > 1:
                #    # ase.io.write(f'solution{count}.vasp', results[i])
                ase.io.write(os.path.join("..", "results", test_name, f'solution{count}_lattice.vasp'), results_ip[i])
                ase.io.write(os.path.join("..", "results", test_name, f'solution{count}_minimised.vasp'), results[i])
                count += 1

    cryst = results[best_idx]
    print("The lowest found energy is ", best_val, "eV")
    # print("The energy per ion is ", best_val/N_atoms, "eV")
    # ase.io.write(os.path.join("..", "results", test_name, "minimal_energy_structure.vasp"), cryst)
    # print("Rerunning GULP, so that gulp.gout would have optimised structure")
    # opt = calc.get_optimizer(cryst)
    # opt.run(fmax=0.05)
    # cryst.get_potential_energy()
    if printing:
        print("Paused, the files can be copied")
        input()

    return best_val


def get_cif_energies(filename, library, format='cif'):
    filedir = root_dir / 'structures/'
    # Path hacks again
    cryst = ase.io.read(os.path.join(".", filedir / filename), format=format, parallel=False)
    calc = GULP(keywords='conp', library=library)
    calc.set(keywords='opti conjugate conp diff comp c6')
    opt = calc.get_optimizer(cryst)
    opt.run(fmax=0.05)
    energy = cryst.get_potential_energy()

    print("The energy of", filename, "is equal to", energy, "eV")

    return energy


def benchmark():
    # Preparing a folder with results

    shutil.rmtree(os.path.join("..", "results"), ignore_errors=True)
    os.mkdir(os.path.join("..", "results"))

    df_summary = pd.DataFrame(columns=['name', 'grid', 'group', 'best_E', 'expected_E', 'time'])

    for i in range(1, 6):
        if settings[f'SrTiO3_{i}']['test']:
            print("\n\n\n========== Predicting SrTiO3 (perovskite structure) ==========")
            print(settings[f'SrTiO3_{i}'])

            SrTiO = Phase('SrTiO')

            multiple = settings[f'SrTiO3_{i}']['multiple']

            ions_count = {'O': 3 * multiple ** 3, 'Sr': 1 * multiple ** 3, 'Ti': 1 * multiple ** 3}

            start = time()
            allocation = Allocate(ions_count, grid_size=settings[f'SrTiO3_{i}']['grid'], cell_size=3.9 * multiple,
                                  phase=SrTiO)

            # The correct symmetry group is 221, supergroup of 195
            results, runtime, _ = allocation.optimize_cube_symmetry_ase(group=settings[f'SrTiO3_{i}']['group'],
                                                                        PoolSolutions=settings[f'SrTiO3_{i}']['top'],
                                                                        TimeLimit=0)

            best_energy = process_results(lib=SrTiO.filedir / 'SrTiO/buck.lib', results=results,
                                          ions_count=ions_count, test_name=f'SrTiO_{i}')

            energy = get_cif_energies(filename='SrTiO3.cif', library=SrTiO.filedir / 'SrTiO/buck.lib')
            if multiple > 1:
                energy = energy * multiple ** 3
                print("For the given multiple it is equal to ", energy, "eV")

            end = time()
            print('It took ', end='')
            print(" %s seconds including IP and data generation" % (end - start))

            df_summary = df_summary.append({'name': f'SrTiO3_{i}', 'grid': settings[f'SrTiO3_{i}']['grid'],
                                            'group': settings[f'SrTiO3_{i}']['group'], 'best_E': best_energy,
                                            'expected_E': energy, 'time': runtime}, ignore_index=True)

    for i in range(1, 4):
        if settings[f'Y2O3_{i}']['test']:
            print("\n\n\n========== Predicting Y2O3 ==========")
            print(settings[f'Y2O3_{i}'])

            YSrTiO = Phase('YSrTiO')

            ions_count = {'O': 48, 'Y': 32}

            start = time()

            allocation = Allocate(ions_count, grid_size=settings[f'Y2O3_{i}']['grid'], cell_size=10.7, phase=YSrTiO)
            # The actual group is 206, 195 is subgroup
            results, runtime, _ = allocation.optimize_cube_symmetry_ase(group=settings[f'Y2O3_{i}']['group'],
                                                                        PoolSolutions=settings[f'Y2O3_{i}']['top'],
                                                                        TimeLimit=0)

            best_energy = process_results(lib=YSrTiO.filedir / 'YSrTiO/buck.lib', results=results,
                                          ions_count=ions_count, test_name=f'Y2O3_{i}')
            energy = get_cif_energies(filename='Y2O3.cif', library=YSrTiO.filedir / 'YSrTiO/buck.lib')

            end = time()
            print('It took ', end='')
            print(" %s seconds including IP and data generation" % (end - start))

            df_summary = df_summary.append({'name': f'Y2O3_{i}', 'grid': settings[f'Y2O3_{i}']['grid'],
                                            'group': settings[f'Y2O3_{i}']['group'], 'best_E': best_energy,
                                            'expected_E': energy, 'time': runtime}, ignore_index=True)

    # Y2Ti2O7
    for i in range(1, 3):
        if settings[f'Y2Ti2O7_{i}']['test']:
            print("\n\n\n========== Predicting Y2Ti2O7 (pyrochlore structure) ==========")
            print(settings[f'Y2Ti2O7_{i}'])

            YSrTiO = Phase('YSrTiO')

            ions_count = {'O': 56, 'Y': 16, 'Ti': 16}

            start = time()

            allocation = Allocate(ions_count, grid_size=settings[f'Y2Ti2O7_{i}']['grid'], cell_size=10.2, phase=YSrTiO)
            results, runtime, _ = allocation.optimize_cube_symmetry_ase(group=settings[f'Y2Ti2O7_{i}']['group'],
                                                                        PoolSolutions=settings[f'Y2Ti2O7_{i}']['top'],
                                                                        TimeLimit=0)

            best_energy = process_results(lib=YSrTiO.filedir / 'YSrTiO/buck.lib', results=results,
                                          ions_count=ions_count, test_name=f'Y2Ti2O7_{i}')
            energy = get_cif_energies(filename='Y2Ti2O7.cif', library=YSrTiO.filedir / 'YSrTiO/buck.lib')

            end = time()
            print('It took ', end='')
            print(" %s seconds" % (end - start))

            df_summary = df_summary.append({'name': f'Y2Ti2O7_{i}', 'grid': settings[f'Y2Ti2O7_{i}']['grid'],
                                            'group': settings[f'Y2Ti2O7_{i}']['group'], 'best_E': best_energy,
                                            'expected_E': energy, 'time': runtime}, ignore_index=True)

    for i in range(1, 5):
        if settings[f'MgAl2O4_{i}']['test']:
            print("\n\n\n========== Predicting MgAl2O4 (spinel structure) ==========")
            print(settings[f'MgAl2O4_{i}'])

            LiMgAlPO = Phase('LiMgAlPO')

            ions_count = {'O': 32, 'Mg': 8, 'Al': 16}

            start = time()

            allocation = Allocate(ions_count, grid_size=settings[f'MgAl2O4_{i}']['grid'], cell_size=8.2, phase=LiMgAlPO)
            # The actual group is 227, is subgroup
            results, runtime, _ = allocation.optimize_cube_symmetry_ase(group=settings[f'MgAl2O4_{i}']['group'],
                                                                        PoolSolutions=settings[f'MgAl2O4_{i}']['top'],
                                                                        TimeLimit=0)

            best_energy = process_results(lib=LiMgAlPO.filedir / 'LiMgAlPO/buck.lib', results=results,
                                          ions_count=ions_count, test_name=f'MgAl2O4_{i}')
            energy = get_cif_energies(filename='MgAl2O4.cif', library=LiMgAlPO.filedir / 'LiMgAlPO/buck.lib')

            end = time()
            print('It took ', end='')
            print(" %s seconds" % (end - start))

            df_summary = df_summary.append({'name': f'MgAl2O4_{i}', 'grid': settings[f'MgAl2O4_{i}']['grid'],
                                            'group': settings[f'MgAl2O4_{i}']['group'], 'best_E': best_energy,
                                            'expected_E': energy, 'time': runtime}, ignore_index=True)

    for i in range(1, 3):
        if settings[f'Ca3Al2Si3O12_{i}']['test']:
            print("\n\n\n========== Predicting Ca3Al2Si3O12 (garnet structure) ==========")
            print(settings[f'Ca3Al2Si3O12_{i}'])

            CaAlSiO = Phase('CaAlSiO')

            z = 8
            ions_count = {'Ca': 3 * z, 'Al': 2 * z, 'Si': 3 * z,
                          'O': 12 * z}  # z=1 gives 20 ions. garnet is z = 8, 160 ions

            start = time()

            allocation = Allocate(ions_count, grid_size=settings[f'Ca3Al2Si3O12_{i}']['grid'],
                                  cell_size=11.9, phase=CaAlSiO)
            # The actual group is 230
            results, runtime, _ = allocation.optimize_cube_symmetry_ase(group=settings[f'Ca3Al2Si3O12_{i}']['group'],
                                                                        PoolSolutions=settings[f'Ca3Al2Si3O12_{i}'][
                                                                            'top'],
                                                                        TimeLimit=0)

            best_energy = process_results(lib=CaAlSiO.filedir / 'CaAlSiO/pedone.lib', results=results,
                                          ions_count=ions_count, test_name=f'Ca3Al2Si3O12_{i}')
            energy = get_cif_energies(filename='Ca3Al2Si3O12.cif', library=CaAlSiO.filedir / 'CaAlSiO/pedone.lib')

            end = time()
            print('It took ', end='')
            print(" %s seconds" % (end - start))

            df_summary = df_summary.append({'name': f'Ca3Al2Si3O12_{i}', 'grid': settings[f'Ca3Al2Si3O12_{i}']['grid'],
                                            'group': settings[f'Ca3Al2Si3O12_{i}']['group'], 'best_E': best_energy,
                                            'expected_E': energy, 'time': runtime}, ignore_index=True)

    with open(os.path.join("..", "results", "summary.txt"), "w+") as f:
        print("Non-heuristic optimisation using Gurobi with subsequent local minimisation:", file=f)
        print(tabulate(df_summary, headers=["Test name", "Discretisation g", "Space group",
                                            "Best energy (eV)", "Target energy (eV)", "IP solution time (sec)"],
                       tablefmt='github', showindex=False), file=f)

    '''
    
    Start of the quantum section
    
    '''

    df_summary = pd.DataFrame(columns=['name', 'dwave', 'best_E', 'expected_E'])

    if settings['quantum_SrO']['test']:
        print("\n\n\n========== Predicting SrO (rocksalt) using quantum annealer ==========")
        SrTiO = Phase('SrTiO')
        multiple = settings['quantum_SrO']['multiple']  # the number of primitive cells per side
        ions_count = {'O': 4 * multiple ** 3, 'Sr': 4 * multiple ** 3}

        start = time()
        allocation = Allocate(ions_count, grid_size=multiple * 2, cell_size=5.2 * multiple, phase=SrTiO)

        best_energy, target_energy = allocation.optimize_qubo(group=settings['quantum_SrO']['group'],
                                                              at_dwave=settings['quantum_SrO']['at_dwave'],
                                                              num_reads=settings['quantum_SrO']['num_reads'],
                                                              infinity_placement=settings['quantum_SrO'][
                                                                  'infinity_placement'],
                                                              infinity_orbit=settings['quantum_SrO']['infinity_orbit'],
                                                              annealing_time=settings['quantum_SrO']['annealing_time'])

        energy = get_cif_energies(filename='SrO.cif', library=SrTiO.filedir / 'SrTiO/buck.lib')

        df_summary = df_summary.append({'name': 'quantum_SrO', 'dwave': settings['quantum_SrO']['at_dwave'],
                                        'best_E': best_energy, 'expected_E': target_energy}, ignore_index=True)

        end = time()
        print('It took ', end='')
        print(" %s seconds" % (end - start))

    if settings['quantum_ZnS']['test']:
        print("\n\n\n========== Predicting ZnS (wurtzite) using quantum annealer ==========")
        ZnS = Phase('ZnS')
        ions_count = {'Zn': 4, 'S': 4}

        start = time()
        allocation = Allocate(ions_count, grid_size=4, cell_size=5.4, phase=ZnS)

        best_energy, target_energy = allocation.optimize_qubo(group=settings['quantum_ZnS']['group'],
                                               at_dwave=settings['quantum_ZnS']['at_dwave'],
                                               num_reads=settings['quantum_ZnS']['num_reads'],
                                               infinity_placement=settings['quantum_ZnS']['infinity_placement'],
                                               infinity_orbit=settings['quantum_ZnS']['infinity_orbit'],
                                               annealing_time=settings['quantum_ZnS']['annealing_time'])

        energy = get_cif_energies(filename='ZnS.cif', library=ZnS.filedir / 'ZnS/buck.lib')

        df_summary = df_summary.append({'name': 'quantum_ZnS', 'dwave': settings['quantum_ZnS']['at_dwave'],
                                        'best_E': best_energy, 'expected_E': target_energy}, ignore_index=True)

        end = time()
        print('It took ', end='')
        print(" %s seconds" % (end - start))

    if settings['quantum_ZrO2']['test']:
        print("\n\n\n========== Predicting ZrO2 (cubic zirconia) using quantum annealer ==========")
        ZrO = Phase('ZrO')
        # multiple = settings['quantum_ZrO2']['multiple']  # the number of primitive cells per side
        ions_count = {'Zr': 4, 'O': 8}

        start = time()
        allocation = Allocate(ions_count, grid_size=4, cell_size=5.07, phase=ZrO)

        best_energy, target_energy = allocation.optimize_qubo(group=settings['quantum_ZrO2']['group'],
                                               at_dwave=settings['quantum_ZrO2']['at_dwave'],
                                               num_reads=settings['quantum_ZrO2']['num_reads'],
                                               infinity_placement=settings['quantum_ZrO2']['infinity_placement'],
                                               infinity_orbit=settings['quantum_ZrO2']['infinity_orbit'],
                                               annealing_time=settings['quantum_ZrO2']['annealing_time'])

        energy = get_cif_energies(filename='ZrO2.cif', library=ZrO.filedir / 'ZrO/buck.lib')

        df_summary = df_summary.append({'name': 'quantum_ZrO2', 'dwave': settings['quantum_ZrO2']['at_dwave'],
                                        'best_E': best_energy, 'expected_E': target_energy}, ignore_index=True)

        end = time()
        print('It took ', end='')
        print(" %s seconds" % (end - start))

    if settings['quantum_SrTiO3']['test']:
        print("\n\n\n========== Predicting SrTiO3 (perovskite) using quantum annealer ==========")
        SrTiO = Phase('SrTiO')
        multiple = settings['quantum_SrTiO3']['multiple']  # the number of primitive cells per side
        ions_count = {'O': 3 * multiple ** 3, 'Sr': 1 * multiple ** 3, 'Ti': 1 * multiple ** 3}

        start = time()
        allocation = Allocate(ions_count, grid_size=multiple * 2, cell_size=3.9 * multiple, phase=SrTiO)

        best_energy, target_energy = allocation.optimize_qubo(group=settings['quantum_SrTiO3']['group'],
                                               at_dwave=settings['quantum_SrTiO3']['at_dwave'],
                                               num_reads=settings['quantum_SrTiO3']['num_reads'],
                                               infinity_placement=settings['quantum_SrTiO3']['infinity_placement'],
                                               infinity_orbit=settings['quantum_SrTiO3']['infinity_orbit'],
                                               annealing_time=settings['quantum_SrTiO3']['annealing_time'])

        energy = get_cif_energies(filename='SrTiO3.cif', library=SrTiO.filedir / 'SrTiO/buck.lib')

        df_summary = df_summary.append({'name': 'quantum_SrTiO3', 'dwave': settings['quantum_SrTiO3']['at_dwave'],
                                        'best_E': best_energy, 'expected_E': target_energy}, ignore_index=True)

        end = time()
        print('It took ', end='')
        print(" %s seconds" % (end - start))

    with open(os.path.join("..", "results", "summary.txt"), "a") as f:
        print("\n\n\n\n\n Quantum annealing for the periodic lattice atom allocation.\n", file=f)
        print(tabulate(df_summary, headers=["Test name", "D-Wave", "Best energy (eV)", "Target energy (eV)"],
                       tablefmt='github', showindex=False), file=f)


if __name__ == "__main__":
    benchmark()
