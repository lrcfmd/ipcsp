import gurobipy as gb
import numpy as np
import os

from ipcsp import root_dir
from ipcsp.matrix_generator import Phase, get_Ewald, get_Buck, get_garnet
import ase
from ipcsp.grids_and_symmetry import cubic
import json
from ipcsp.lp_to_bqm import BQM

griddir = root_dir / 'data/grids/'

class Allocate:

    def __init__(self, ions_count, grid_size, cell_size, phase):

        self.ions = ions_count

        self.phase = phase

        self.grid = grid_size

        self.cell = cell_size

        self.model = None

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def solution_to_Atoms(self, i, orbits):

        self.model.params.SolutionNumber = i

        grid_positions = cubic(self.grid)
        # ions = list(self.ions.keys())

        # Atoms
        symbols = ''
        positions = []

        for v in self.model.getVars():
            if v.Xn == 1:
                # print(v.varName, end=' ')
                t, o = v.varName.split(sep='_')

                # The element itself
                positions.append(grid_positions[int(o)])
                symbols += t

                # And its orbit
                for pos in orbits[o]:
                    positions.append(grid_positions[pos])
                    symbols += t

        # print(symbols, positions)
        return ase.Atoms(symbols=symbols, scaled_positions=positions,
                         cell=[self.cell, self.cell, self.cell], pbc=True)

    def optimize_cube_symmetry_ase(self, group='1', PoolSolutions=1, TimeLimit=0, verbose=True):
        '''
        The function to generate an integer program and solve allocation problem using Gurobi.
        We rely on atomic simulation environment to handle allocations afterwards.
        '''

        N = self.grid ** 3  # number
        T = len(self.ions)  # different types

        # PATH hack
        with open(os.path.join(".", griddir / 'CO{grid}G{group}.json'.format(grid=self.grid, group=group)), "r") as f:
            orbits = json.load(f)

        orb_key = list(orbits.keys())

        if verbose:
            #print("Orbits", orbits)
            #print("Orbit keys", orb_key)

            #print("Orbits and their sizes. Similar to asymmetric units. Given as representative position : count.\n")
            #for k in orb_key:
            #    print(k, ':', len(orbits[k]) + 1, end='; ')
            #print('\n')
            print("Generating integer program\n")

        orb_size = [len(orbits[k]) + 1 for k in orb_key]

        # o_pos[position] = number_of_the_orbit
        o_pos = []
        for i in range(N):
            for orb, pos in orbits.items():
                if int(orb) == i:
                    o_pos.append(orb_key.index(orb))
                    break

                if i in pos:
                    o_pos.append(orb_key.index(orb))
                    break

        # print(orbits)
        # print(orb_key)
        # print(o_pos)
        # print(len(o_pos))

        O = len(orbits)
        types = list(self.ions.keys())  # ordered list of elements
        counts = [self.ions[t] for t in types]

        m = gb.Model('Ion allocation in {name} with symmetry {group}'.format(name=self.phase.name, group=group))
        Vars = [[] for i in range(T)]

        # Create variables
        for i in range(O):  # iterate over all orbitals
            tmp_var = []
            for j in range(T):
                Vars[j] += [m.addVar(vtype=gb.GRB.BINARY, name=str(types[j]) + '_' + orb_key[i])]
                tmp_var += [Vars[j][-1]]
            if i == 0:
                m.addConstr(gb.LinExpr([1.0] * T, tmp_var) == 1, 'first_orbit_has_ion')
                # m.addConstr(gb.LinExpr([1.0], [Vars[j][0]]) == 1, 'first_orbit_has_Ti')
            else:
                m.addConstr(gb.LinExpr([1.0] * T, tmp_var) <= 1, f'one_per_orbit_{i}')

        for j in range(T):
            tmp = gb.LinExpr()
            for i in range(O):
                tmp.add(Vars[j][i], orb_size[i])
            m.addConstr(tmp == counts[j], f"number_of_ions_type_{j}")

        print("Variables and constraints were generated")
        energy = gb.QuadExpr()

        # Coulomb interaction
        dist = get_Ewald(self.grid, self.cell)
        # np.savetxt('testEwaldNew.out', dist, delimiter=',')

        for i1 in range(N):
            # print(i1)

            for j1 in range(T):  # self-interaction
                energy.add(Vars[j1][o_pos[i1]] * Vars[j1][o_pos[i1]] * dist[i1, i1] * self.phase.charge[types[j1]] ** 2)
                # energy.add(pt.Lattice.atom_charge[quantity[j1][0]]*pt.Lattice.atom_charge[quantity[j1][0]]*Vars[j1][i1]*Vars[j1][i1]*dist[i1,i1])

            for i2 in range(i1 + 1, N):
                # print(i2)
                for j1 in range(T):  # pairwise Coulumb
                    energy.add(Vars[j1][o_pos[i1]] * Vars[j1][o_pos[i2]] * 2 * dist[i1, i2] * self.phase.charge[
                        types[j1]] ** 2)  # i1,i2 have the same type of ion
                    # old: energy.add(pt.Lattice.atom_charge[quantity[j1][0]]*pt.Lattice.atom_charge[quantity[j1][0]]*Vars[j1][i1]*Vars[j1][i2]*2*dist[i1,i2]) #i1,i2 have the same type of ion

                    for j2 in range(j1 + 1, T):
                        energy.add(Vars[j1][o_pos[i1]] * Vars[j2][o_pos[i2]] * 2 * dist[i1, i2] * self.phase.charge[
                            types[j1]] * self.phase.charge[types[j2]])  # Two different types
                        energy.add(Vars[j2][o_pos[i1]] * Vars[j1][o_pos[i2]] * 2 * dist[i1, i2] * self.phase.charge[
                            types[j1]] * self.phase.charge[types[j2]])  # Symmetrical case
                        # size+=2
        # print(dist)
        del dist
        print("Ewald sum contribution was added to the objective function")

        # HACK for the garnet
        if self.phase.garnet:
            print("Generating Morse and Leonard parts")
            # Morse and Leonard parts
            for ion_pair in self.phase.gar_param:

                if not ((ion_pair[0] in types) and (ion_pair[1] in types)):
                    continue

                # I am keeping the old name, too lazy to rename
                buck = get_garnet(ion_pair, self.grid, self.cell, self.phase)

                if ion_pair[0] == ion_pair[1]:
                    j1 = types.index(ion_pair[0])

                    for i1 in range(N):
                        for i2 in range(i1, N):
                            # if i1==58 and i2 == 59:
                            #     print(buck[i1, i2])
                            energy.add(Vars[j1][o_pos[i1]] * Vars[j1][o_pos[i2]] * buck[i1, i2])

                else:

                    j1 = types.index(ion_pair[0])
                    j2 = types.index(ion_pair[1])

                    for i1 in range(N):
                        # TODO: Two different ions at the same position is impossible, should we put MAX here?
                        # energy.add(Vars[j1][i1]*Vars[j1][i1]*buck[i1, i1])

                        for i2 in range(i1 + 1, N):
                            energy.add(Vars[j1][o_pos[i1]] * Vars[j2][o_pos[i2]] * buck[i1, i2])
                            energy.add(Vars[j2][o_pos[i1]] * Vars[j1][o_pos[i2]] * buck[i1, i2])

                del buck

        else:

            # Buckingham Part
            for ion_pair in self.phase.buck:

                if not ((ion_pair[0] in types) and (ion_pair[1] in types)):
                    continue

                buck = get_Buck(ion_pair, self.grid, self.cell, self.phase)

                if ion_pair[0] == ion_pair[1]:
                    j1 = types.index(ion_pair[0])

                    for i1 in range(N):
                        for i2 in range(i1, N):
                            # if i1==58 and i2 == 59:
                            #     print(buck[i1, i2])
                            energy.add(Vars[j1][o_pos[i1]] * Vars[j1][o_pos[i2]] * buck[i1, i2])

                else:

                    j1 = types.index(ion_pair[0])
                    j2 = types.index(ion_pair[1])

                    for i1 in range(N):
                        # TODO: Two different ions at the same position is impossible, should we put MAX here?
                        # energy.add(Vars[j1][i1]*Vars[j1][i1]*buck[i1, i1])

                        for i2 in range(i1 + 1, N):
                            energy.add(Vars[j1][o_pos[i1]] * Vars[j2][o_pos[i2]] * buck[i1, i2])
                            energy.add(Vars[j2][o_pos[i1]] * Vars[j1][o_pos[i2]] * buck[i1, i2])

                del buck

        print("Objective function was generated")

        m.setObjective(energy, gb.GRB.MINIMIZE)
        self.model = m

        if TimeLimit > 0:
            m.params.TimeLimit = TimeLimit

        if not verbose:
            m.params.OutputFlag = 0

        if PoolSolutions > 1:
            m.params.PoolSolutions = PoolSolutions
            m.params.PoolSearchMode = 2

        m.Params.NodefileStart = 1

        # mp = m.presolve()
        print("Writing model file")
        m.write("model.lp")

        m.optimize()

        if m.status == gb.GRB.CUTOFF:
            print("Cutoff! No solution with negative energy.")
            return None

        # if m.status == gb.GRB.TIME_LIMIT:
        #     if m.objVal < 0:
        #         print("Time limit reached. There is a solution")
        #     return None

        if m.status == gb.GRB.OPTIMAL or m.status == gb.GRB.TIME_LIMIT or gb.GRB.INTERRUPTED:
            print("There are", m.SolCount, "solutions")
            res = []
            for i in range(m.SolCount):
                res.append(self.solution_to_Atoms(i, orbits))

            print("\nThe optimal assignment is as follows:")
            for v in m.getVars():
                if v.x == 1:
                    print(v.varName, end=' ')
                # print('%s %g' % (v.varName, v.x))
            print()
            print('Minimal energy via optimizer: %g' % m.objVal)

            if PoolSolutions > 1:
                return res
            else:
                return res[0:1]

        return None

    def optimize_qubo(self, group='1', at_dwave=False, num_reads=10,
                             infinity_placement=100, infinity_orbit=100, annealing_time=200):
        """
        The function to optimise the structure on the quantum annealer
        infinity_orbit is the penalty for putting two ions on the orbit
        infinity_placement is the penalty for placing the incorrect number of ions into the structure
        """
        import neal
        import dimod
        import dwave.embedding
        import dwave.system
        from dwave.system.samplers import DWaveSampler
        from dwave.system.composites import EmbeddingComposite

        print("Running integer programming optimisation to generate a model file with the required coefficients and "
              "obtain the ground truth for the lowest energy allocation.")

        self.optimize_cube_symmetry_ase(group=group, verbose=False)

        print("Generating quadratic unconstrained binary problem from model.lp")

        bqm_model = BQM()
        bqm_model.parse_lp("model.lp")
        bqm_model.parse_constraints()
        bqm_model.max_bound()
        bqm_model.qubofy(infinity_placement, infinity_orbit)

        np.set_printoptions(suppress=True)
        print('Five number summary of the interaction coefficients of the Ising hamiltonian:', np.percentile(
            np.array(list(bqm_model.quadratic.values())), [0, 25, 50, 75, 100], interpolation='midpoint'))

        # print(bqm_model.linear, bqm_model.quadratic, bqm_model.offset)
        print("The offset is equal to", bqm_model.offset)

        # solver = neal.SimulatedAnnealingSampler()
        bqm = dimod.BinaryQuadraticModel(bqm_model.linear, bqm_model.quadratic, bqm_model.offset, dimod.BINARY)

        # bqm.fix_variable(('Sr', 7), 1)
        # bqm.fix_variable(('Ti', 0), 1)
        # print(list(bqm.variables))
        print("There are ", len(bqm.variables), "variables in the program")
        print("Running the Annealer")

        def stoic(datum):
            # counts = {'O': 0, 'Sr': 0, 'Ti': 0}
            counts = {}
            for k, v in datum[0].items():
                # counts[k[0]] += v
                if k[0] in counts:
                    counts[k[0]] += v
                else:
                    counts[k[0]] = v
            # print('===', datum[0])
            return 'Counts: ' + str(counts)

        def simplify(datum):
            sample = {'energy': 0, 'sample': [], 'num_occurrences': 0}
            sample['energy'] = datum[1]
            sample['num_occurrences'] = int(datum[2])

            for k, v in datum[0].items():
                if v == 1:
                    sample['sample'].append(k)

            # print('<', sample, '>')
            # print('===', datum[0])
            return sample

        if at_dwave:
            embedding = dwave.embedding.chimera.find_clique_embedding(len(bqm.variables), 16)
            print("The number of qubits: ", sum(len(chain) for chain in embedding.values()))
            print("The longest chain: ", max(len(chain) for chain in embedding.values()))
            exit()
            sampler = EmbeddingComposite(DWaveSampler())
            response = sampler.sample(bqm, num_reads=num_reads, annealing_time=annealing_time)
            min_energy = 1000000
            sol = None
            json_result = []

            for datum in response.data(['sample', 'energy', 'num_occurrences']):
                print(simplify(datum), stoic(datum))
                json_result.append(simplify(datum))

                if datum.energy < min_energy:
                    sol = datum
                    min_energy = datum.energy
            # print(type(sol))

            with open('last_dwave.json', 'w') as f:
                json.dump(json_result, f, indent=2)

            for i in sol.sample.keys():
                if sol.sample[i] == 1:
                    print(i)
            print("Energy: ", sol.energy, "Occurrences: ", sol.num_occurrences)
        else:
            solver = neal.SimulatedAnnealingSampler()
            response = solver.sample(bqm, num_reads=num_reads)
            min_energy = 10000
            sample = 0
            for datum in response.data(['sample', 'energy', 'num_occurrences']):
                print(simplify(datum), stoic(datum))

                if datum.energy < min_energy:
                    sample = datum
                    min_energy = datum.energy

            for i in sample.sample.keys():
                if sample.sample[i] == 1:
                    print(i)
            print("Energy: ", sample.energy, "Occurrences: ", sample.num_occurrences)
