"""
The file to run multiple crystal structure predictions at the same time with more or less universal settings.
1. SrTiO3 for different Z (perovskite structure)
2. Y2O3 (bixbyite structure)
3. Y2Ti2O7 (pyrochlore structure)
4. MgAl2O4 (spinel structure)
5. Ca3Al2Si3O12 (garnet structure)

Quantum experiments for SrO, SrTiO3, ZnS, ZrO2
"""

from time import time
from integer_program import Allocate
from matrix_generator import Phase
from ase.calculators.gulp import GULP
import ase.io

# The parameters to reproduce Table 1 in the paper and quantum experiments
settings = {# perovskite structure
            'SrTiO_1': {'test': True, 'multiple': 1, 'group': 1, 'top': 1, 'grid': 4},  # group is 221, sub 195
            'SrTiO_2': {'test': False, 'multiple': 2, 'group': 195, 'top': 1, 'grid': 8},  # group is 221, sub 195
            'SrTiO_3': {'test': False, 'multiple': 3, 'group': 221, 'top': 1, 'grid': 6},  # group is 221, sub 195
            'SrTiO_4': {'test': False, 'multiple': 3, 'group': 200, 'top': 1, 'grid': 6},  # group is 221, sub 195
            'SrTiO_5': {'test': False, 'multiple': 3, 'group': 195, 'top': 1, 'grid': 6},  # group is 221, sub 195
            # bixbyite structure
            'Y2O3_1': {'test': True, 'group': 206, 'top': 1, 'grid': 8},  # group is 206, sub 199, sub 198
            'Y2O3_2': {'test': False, 'group': 199, 'top': 1, 'grid': 8},  # group is 206, sub 199, sub 198
            'Y2O3_3': {'test': False, 'group': 206, 'top': 1, 'grid': 16},  # group is 206, sub 199, sub 198
            # pyrochlore structure of Y2Ti2O7
            'pyro_1': {'test': True, 'group': 227, 'top': 5, 'grid': 8},  # group is 227, 196, 195
            'pyro_2': {'test': False, 'group': 227, 'top': 1, 'grid': 16},  # group is 227, 196, 195
            # spinel structure of MgAl2O4
            'spinel_1': {'test': True, 'group': 227, 'top': 1, 'grid': 8},  # group is 227, sub 195, 196, grid is 8
            'spinel_2': {'test': False, 'group': 227, 'top': 1, 'grid': 16},  # group is 227, sub 195, 196, grid is 8
            'spinel_3': {'test': False, 'group': 196, 'top': 1, 'grid': 8},  # group is 227, sub 195, 196, grid is 8
            'spinel_4': {'test': False, 'group': 195, 'top': 10, 'grid': 8},  # group is 227, sub 195, 196, grid is 8
            # garnet structure of Ca3Al2Si3O12
            'garnet_1': {'test': True, 'group': 230, 'top': 1, 'grid': 16},  # group is 230, sub 206, 199, grid is 8
            'garnet_2': {'test': False, 'group': 206, 'top': 10, 'grid': 8},  # group is 230, sub 206, 199, grid is 8
            # Quantum experiments
            # 'at_dwave': True will connect to the quantum annealer and use your computational budget
            # 'at_dwave': False will rely on local simulated annealing
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


def process_results(lib, results, printing=False):

    calc = GULP(keywords='conp', library=lib)

    # Compute the number of atoms
    N_atoms = 0
    for k, v in ions_count.items():
        N_atoms += v

    init = [0] * len(results)
    final = [0] * len(results)
    best_val = 0
    best_idx = 0
    ase.io.write("best_ipcsp.vasp", results[0])
    print("Processing and locally optimising solutions from the integer program\n")
    for idx, cryst in enumerate(results):
        if len(cryst.arrays['positions']) == N_atoms:
            cryst.calc = calc
            init[idx] = cryst.get_potential_energy()
            print("Initial:", init[idx])
        else:
            print("GULP recieved bad solution")

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
                #print("Final:", final[idx])
                # input()
            # print("Energy initial: ", cryst.get_potential_energy(), " final: ", final)

    count = 1
    for i in range(len(results)):
        if final[i] != 0:
            print(count, "Energy initial: ", init[i], " final: ", final[i])
            if printing:
                ase.io.write(f'solution{count}.vasp', results[i])
            count += 1

    cryst = results[best_idx]
    print("The lowest found energy is ", best_val, "eV")
    # print("The energy per ion is ", best_val/N_atoms, "eV")
    ase.io.write("best.vasp", cryst)
    # print("Rerunning GULP, so that gulp.gout would have optimised structure")
    # opt = calc.get_optimizer(cryst)
    # opt.run(fmax=0.05)
    # cryst.get_potential_energy()
    if printing:
        print("Paused, the files can be copied")
        input()


def get_cif_energies(filename, library, format='cif'):
    filedir = './structures/'
    cryst = ase.io.read(filedir + filename, format=format, parallel=False)
    calc = GULP(keywords='conp', library=library)
    calc.set(keywords='opti conjugate conp diff comp c6')
    opt = calc.get_optimizer(cryst)
    opt.run(fmax=0.05)
    energy = cryst.get_potential_energy()

    print("The energy of", filename, "is equal to", energy, "eV")

    return energy


if __name__ == "__main__":
    for i in range(1, 6):
        if settings[f'SrTiO_{i}']['test']:
            print("========== Predicting SrTiO3 (perovskite structure) ==========")
            print(settings[f'SrTiO_{i}'])

            SrTiO = Phase('SrTiO')

            multiple = settings[f'SrTiO_{i}']['multiple']

            ions_count = {'O': 3 * multiple ** 3, 'Sr': 1 * multiple ** 3, 'Ti': 1 * multiple ** 3}

            start = time()
            allocation = Allocate(ions_count, grid_size=settings[f'SrTiO_{i}']['grid'], cell_size=3.9 * multiple, phase=SrTiO)

            # The correct symmetry group is 221, supergroup of 195
            results = allocation.optimize_cube_symmetry_ase(group=settings[f'SrTiO_{i}']['group'],
                                                            PoolSolutions=settings[f'SrTiO_{i}']['top'], TimeLimit=0)

            process_results(lib=SrTiO.filedir + 'SrTiO/buck.lib', results=results)
            energy = get_cif_energies(filename='SrTiO3.cif', library=SrTiO.filedir + 'SrTiO/buck.lib')
            if multiple > 1:
                print("For the given multiple it is equal to ", energy * multiple ** 3, "eV")

            end = time()
            print('It took ', end='')
            print(" %s seconds including IP and data generation" % (end - start))

    for i in range(1, 6):
        if settings[f'Y2O3_{i}']['test']:
            print("========== Predicting Y2O3 ==========")
            print(settings[f'Y2O3_{i}'])

            YSrTiO = Phase('YSrTiO')

            ions_count = {'O': 48, 'Y': 32}

            start = time()

            allocation = Allocate(ions_count, grid_size=settings[f'Y2O3_{i}']['grid'], cell_size=10.7, phase=YSrTiO)
            # The actual group is 206, 195 is subgroup
            results = allocation.optimize_cube_symmetry_ase(group=settings[f'Y2O3_{i}']['group'],
                                                            PoolSolutions=settings[f'Y2O3_{i}']['top'], TimeLimit=0)

            process_results(lib=YSrTiO.filedir + 'YSrTiO/buck.lib', results=results)
            get_cif_energies(filename='Y2O3.cif', library=YSrTiO.filedir + 'YSrTiO/buck.lib')

            end = time()
            print('It took ', end='')
            print(" %s seconds including IP and data generation" % (end - start))

    # Y2Ti2O7
    for i in range(1, 6):
        if settings[f'pyro_{i}']['test']:
            print("========== Predicting Y2Ti2O7 (pyrochlore structure) ==========")
            print(settings[f'pyro_{i}'])

            YSrTiO = Phase('YSrTiO')

            ions_count = {'O': 56, 'Y': 16, 'Ti': 16}

            start = time()

            allocation = Allocate(ions_count, grid_size=settings[f'pyro_{i}']['grid'], cell_size=10.2, phase=YSrTiO)
            results = allocation.optimize_cube_symmetry_ase(group=settings[f'pyro_{i}']['group'],
                                                            PoolSolutions=settings[f'pyro_{i}']['top'], TimeLimit=0)

            process_results(lib=YSrTiO.filedir + 'YSrTiO/buck.lib', results=results)
            get_cif_energies(filename='Y2Ti2O7.cif', library=YSrTiO.filedir + 'YSrTiO/buck.lib')

            end = time()
            print('It took ', end='')
            print(" %s seconds" % (end - start))

    for i in range(1, 6):
        if settings[f'spinel_{i}']['test']:
            print("========== Predicting MgAl2O4 (spinel structure) ==========")
            print(settings[f'spinel_{i}'])

            LiMgAlPO = Phase('LiMgAlPO')

            ions_count = {'O': 32, 'Mg': 8, 'Al': 16}

            start = time()

            allocation = Allocate(ions_count, grid_size=settings[f'spinel_{i}']['grid'], cell_size=8.2, phase=LiMgAlPO)
            # The actual group is 227, is subgroup
            results = allocation.optimize_cube_symmetry_ase(group=settings[f'spinel_{i}']['group'],
                                                            PoolSolutions=settings[f'spinel_{i}']['top'], TimeLimit=0)

            process_results(lib=LiMgAlPO.filedir + 'LiMgAlPO/buck.lib', results=results)
            get_cif_energies(filename='MgAl2O4.cif', library=LiMgAlPO.filedir + 'LiMgAlPO/buck.lib')

            end = time()
            print('It took ', end='')
            print(" %s seconds" % (end - start))

    for i in range(1, 6):
        if settings[f'garnet__{i}']['test']:
            print("========== Predicting Ca3Al2Si3O12 (garnet structure) ==========")
            print(settings[f'garnet_{i}'])

            CaAlSiO = Phase('CaAlSiO')

            z = 8
            ions_count = {'Ca': 3 * z, 'Al': 2 * z, 'Si': 3 * z, 'O': 12 * z}  # z=1 gives 20 ions. garnet is z = 8, 160 ions

            start = time()

            allocation = Allocate(ions_count, grid_size=settings[f'garnet_{i}']['grid'], cell_size=11.9, phase=CaAlSiO)
            # The actual group is 230
            results = allocation.optimize_cube_symmetry_ase(group=settings[f'garnet_{i}']['group'],
                                                            PoolSolutions=settings[f'garnet_{i}']['top'], TimeLimit=0)

            process_results(lib=CaAlSiO.filedir + 'CaAlSiO/pedone.lib', results=results)
            get_cif_energies(filename='Ca3Al2Si3O12.cif', library=CaAlSiO.filedir + 'CaAlSiO/pedone.lib')

            end = time()
            print('It took ', end='')
            print(" %s seconds" % (end - start))

    if settings['quantum_SrO']['test']:
        print("========== Predicting SrO (rocksalt) using quantum annealer ==========")
        SrTiO = Phase('SrTiO')
        multiple = settings['Quantum_SrTiO']['multiple']  # the number of primitive cells per side
        ions_count = {'O': 4 * multiple ** 3, 'Sr': 4 * multiple ** 3}

        start = time()
        allocation = Allocate(ions_count, grid_size=multiple * 2, cell_size=5.2 * multiple, phase=SrTiO)

        results = allocation.optimize_cube_dict_2(group=settings['quantum_SrO']['group'],
                                                  at_dwave=settings['quantum_SrO']['at_dwave'],
                                                  num_reads=settings['quantum_SrO']['num_reads'],
                                                  infinity_placement=settings['quantum_SrO']['infinity_placement'],
                                                  infinity_orbit=settings['quantum_SrO']['infinity_orbit'],
                                                  annealing_time=settings['quantum_SrO']['annealing_time'])

        get_cif_energies(filename='SrO.cif', library=SrTiO.filedir + 'SrTiO/buck.lib')

        end = time()
        print('It took ', end='')
        print(" %s seconds" % (end - start))

    if settings['quantum_ZnS']['test']:
        print("========== Predicting ZnS (wurtzite) using quantum annealer ==========")
        ZnS = Phase('ZnS')
        ions_count = {'Zn': 4, 'S': 4}

        start = time()
        allocation = Allocate(ions_count, grid_size=4, cell_size=5.4, phase=ZnS)

        results = allocation.optimize_cube_dict_2(group=settings['quantum_ZnS']['group'],
                                                  at_dwave=settings['quantum_ZnS']['at_dwave'],
                                                  num_reads=settings['quantum_ZnS']['num_reads'],
                                                  infinity_placement=settings['quantum_ZnS']['infinity_placement'],
                                                  infinity_orbit=settings['quantum_ZnS']['infinity_orbit'],
                                                  annealing_time=settings['quantum_ZnS']['annealing_time'])

        get_cif_energies(filename='ZnS.cif', library=ZnS.filedir + 'ZnS/buck.lib')

        end = time()
        print('It took ', end='')
        print(" %s seconds" % (end - start))

    if settings['quantum_ZrO2']['test']:
        print("========== Predicting ZrO2 (cubic zirconia) using quantum annealer ==========")
        ZrO = Phase('ZrO')
        multiple = settings['quantum_ZrO2']['multiple']  # the number of primitive cells per side
        ions_count = {'Zr': 4, 'O': 8}

        start = time()
        allocation = Allocate(ions_count, grid_size=4, cell_size=5.07, phase=ZrO)

        results = allocation.optimize_cube_dict_2(group=settings['quantum_ZrO2']['group'],
                                                  at_dwave=settings['quantum_ZrO2']['at_dwave'],
                                                  num_reads=settings['quantum_ZrO2']['num_reads'],
                                                  infinity_placement=settings['quantum_ZrO2']['infinity_placement'],
                                                  infinity_orbit=settings['quantum_ZrO2']['infinity_orbit'],
                                                  annealing_time=settings['quantum_ZrO2']['annealing_time'])

        get_cif_energies(filename='ZrO2.cif', library=ZrO.filedir + 'ZrO/buck.lib')

        end = time()
        print('It took ', end='')
        print(" %s seconds" % (end - start))

    if settings['quantum_SrTiO3']['test']:
        print("========== Predicting SrTiO3 or other using quantum annealer ==========")
        SrTiO = Phase('SrTiO')
        multiple = settings['quantum_SrTiO3']['multiple']  # the number of primitive cells per side
        ions_count = {'O': 3 * multiple ** 3, 'Sr': 1 * multiple ** 3, 'Ti': 1 * multiple ** 3}

        start = time()
        allocation = Allocate(ions_count, grid_size=multiple * 2, cell_size=3.9 * multiple, phase=SrTiO)

        results = allocation.optimize_cube_dict_2(group=settings['quantum_SrTiO3']['group'],
                                                  at_dwave=settings['quantum_SrTiO3']['at_dwave'],
                                                  num_reads=settings['quantum_SrTiO3']['num_reads'],
                                                  infinity_placement=settings['quantum_SrTiO3']['infinity_placement'],
                                                  infinity_orbit=settings['quantum_SrTiO3']['infinity_orbit'],
                                                  annealing_time=settings['quantum_SrTiO3']['annealing_time'])

        get_cif_energies(filename='SrTiO3.cif', library=SrTiO.filedir + 'SrTiO/buck.lib')

        end = time()
        print('It took ', end='')
        print(" %s seconds" % (end - start))
