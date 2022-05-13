"""
A routine to convert quadratic binary programs with linear constraints arising
during CSP to to Quadratic Unconstraint Binary Optimisation problems
suitable for Quantum Annealers

IP Model is represented in lp format
and implementation is CSP specific
"""

import re

DEBUG = False


class BQM:

    p_ion = re.compile(r'(?P<specie>\S+)_(?P<pos>\d+)')
    p_rhs = re.compile(r'(=|<=)\s*(?P<int>\d+)')

    def __init__(self):
        self.linear = {}
        self.quadratic = {}
        self.offset = 0
        self.variables = []
        self.constraints_str = []

        # constraints are kept as dict with the keys:
        # 'type' is LEQ or EQ
        # 'lhs' [(coefficient, (specie, pos))]
        #  'rhs' integer
        self.constraints_dict = []

    def parse_lp(self, filename, mult=0.5):
        """
        mult deals with the /2 in the energy representation
        """

        ''' Old version that can't handle the engineering notation
        p_square = re.compile(r'(?P<coeff>[\+-]?\s*\d*\.?\d+)\s*(?P<var>\S+_\d+)\s*\^2')
        p_product = re.compile(r'(?P<coeff>[\+-]?\s*\d*\.?\d+)\s*(?P<var_1>\S+_\d+)\s*\*\s*(?P<var_2>\S+_\d+)')
        # p_ion = re.compile(r'(?P<specie>\S+)_(?P<pos>\d+)')
        '''
        p_square = re.compile(r'(?P<coeff>[-+]?\s*(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*(?P<var>\S+_\d+)\s*\^2')
        p_product = re.compile(r'(?P<coeff>[-+]?\s*(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*(?P<var_1>\S+_\d+)\s*\*\s*('
                               r'?P<var_2>\S+_\d+)')
        with open(filename) as f:

            line = ""

            # Skipping the beginning till the object function
            while not line.startswith("Minimize"):
                line = f.readline()

            # Parse till we hit the constraints
            line = f.readline()
            while not line.startswith("Subject To"):
                # squares = p_square.findall(line)
                squares = [m.groupdict() for m in p_square.finditer(line)]
                if squares:
                    # print(squares)
                    for match in squares:
                        energy, name = match['coeff'], match['var']
                        # m = BQM.p_ion.search(name)
                        # k = (m.group('specie'), int(m.group('pos')))
                        k = BQM.specie_pos(name)

                        if k in self.linear:
                            print(f"Another quadratic term for {name} was encountered")
                        else:
                            self.linear[k] = float(energy.replace(' ', ''))
                            # print(k, float(energy.replace(' ', '')))
                            if name not in self.variables:
                                self.variables.append(name)

                # products = p_product.findall(line)
                products = [m.groupdict() for m in p_product.finditer(line)]
                if products:
                    # print(products)
                    for match in products:
                        # energy, name_1, name_2 = match
                        energy, name_1, name_2 = match['coeff'], match['var_1'], match['var_2']
                        # m1 = BQM.p_ion.search(name_1)
                        # m2 = BQM.p_ion.search(name_2)
                        #
                        # if name_1 == 'Y_95':
                        #     print("Hit")

                        t1 = BQM.specie_pos(name_1)
                        t2 = BQM.specie_pos(name_2)

                        k = BQM.order(t1, t2)

                        # k = None
                        # # dealing with ordering mess, first compare positions, then compare species
                        # # go slowly and systematically
                        # swap = False
                        #
                        # if m1.group('pos') == m2.group('pos'):
                        #     if m1.group('specie') > m2.group('specie'):
                        #         swap = True
                        #
                        # if m1.group('pos') > m2.group('pos'):
                        #     swap = True
                        #
                        # if swap:
                        #     m1, m2 = m2, m1
                        #
                        # k = ((m1.group('specie'), int(m1.group('pos'))), (m2.group('specie'), int(m2.group('pos'))))

                        if k in self.quadratic:
                            print(f"Another quadratic term for {name_1} and {name_2} was encountered")
                        else:
                            self.quadratic[k] = float(energy.replace(' ', ''))
                            # print(k, float(energy.replace(' ', '')))
                            if name_1 not in self.variables:
                                self.variables.append(name_1)
                            if name_2 not in self.variables:
                                self.variables.append(name_2)

                line = f.readline()

            # Parse the constraints till we hit bounds
            # Since the constraints are can be multiline,
            # we glue the lines on the go and then process them
            # we assume that all variables are binary afterwards
            line = f.readline()  # it either has first constraint or "Bounds"
            line_prev = line.strip()
            while not line.startswith("Bounds"):

                if ':' in line:
                    self.constraints_str.append(line_prev)
                    line_prev = line.strip()
                else:
                    line_prev += line.strip()

                line = f.readline()

            if not line_prev.startswith("Bounds"):
                self.constraints_str.append(line_prev)

            if len(self.constraints_str) > 1:
                self.constraints_str = self.constraints_str[1:]

            print(len(self.variables), "Variables in the binary program")
            # print(self.constraints_str[0:4])
            # print(self.constraints_str[-2:])

        if mult != 1:
            for k in self.quadratic:
                self.quadratic[k] = self.quadratic[k] * mult

            for k in self.linear:
                self.linear[k] = self.linear[k] * mult

    def parse_constraints(self):
        for constraint in self.constraints_str:
            dict_con = {}
            if '<=' in constraint:
                dict_con['type'] = 'LEQ'
            elif '=' in constraint:
                dict_con['type'] = 'EQ'
            else:
                print(f"Skipping constraint of unknown type {constraint}")

            m = BQM.p_rhs.search(constraint)
            dict_con['rhs'] = int(m.group('int'))

            # Getting the variables with coefficients
            dict_con['lhs'] = []
            for var in self.variables:

                p_var = re.compile(r'(?P<coeff>\d+)?\s*' + var + r'\D')
                m = p_var.search(constraint)

                if m:
                    if m.group('coeff') is not None:
                        dict_con['lhs'].append((int(m.group('coeff')), BQM.specie_pos(var)))
                    else:
                        dict_con['lhs'].append((1, BQM.specie_pos(var)))

            self.constraints_dict.append(dict_con)

    def max_bound(self, max=None):
        """
        bound the positive coefficients, that tend to
        become very large for high symmetry structures
        """

        # print(f"Before {self.linear}")
        # print(f"Before {self.quadratic}")
        if max is None:
            lowest = min([v for k, v in self.linear.items()])
            lowest = min(lowest, min([v for k, v in self.quadratic.items()]))

            # max = -3*lowest
            max = -lowest

        for k, v in self.linear.items():
            self.linear[k] = min(self.linear[k], max)

        for k, v in self.quadratic.items():
            self.quadratic[k] = min(self.quadratic[k], max)

        # print(f"After {self.linear}")
        # print(f"After {self.quadratic}")
        print(f"The maximum coefficient was set to {max}")

    def qubofy(self, eq_inf, leq_infinity):
        """
        The procedure to get rid of constraints
        Not the general approach,
        here only <= 1 and = constraints are assumed!

        Penalties for breaking constraints
        """
        if DEBUG:
            print(len(self.linear) + len(self.quadratic), " terms")
            print(len(self.constraints_dict), " constraints")
            print(self.constraints_dict)

        for dict_con in self.constraints_dict:
            if dict_con['type'] == 'LEQ':

                if DEBUG:
                    print("=====================", dict_con)

                # I know, you'd forget
                if dict_con['rhs'] != 1:
                    print("Encountered unsupported constraint! Treating RHS as 1")

                N = len(dict_con['lhs'])
                for i in range(N):
                    for j in range(i + 1, N):
                        pair = BQM.order(dict_con['lhs'][i][1], dict_con['lhs'][j][1])
                        if pair in self.quadratic:
                            if DEBUG:
                                print(pair, "added", leq_infinity, "to", self.quadratic[pair])
                            self.quadratic[pair] += leq_infinity
                        else:
                            self.quadratic[pair] = leq_infinity
                            if DEBUG:
                                print("Adding new quadratic term for orbits", pair)
                                print(pair, leq_infinity)
            elif dict_con['type'] == 'EQ':
                if DEBUG:
                    print("=====================", dict_con)
                N = len(dict_con['lhs'])
                vars = dict_con['lhs']
                # Terms involving the rhs constant
                self.offset += eq_inf * dict_con['rhs'] ** 2
                if DEBUG:
                    print(f"Offset is increased by {eq_inf} * {dict_con['rhs']} ** 2")
                for i in range(N):
                    if vars[i][1] in self.linear:
                        if DEBUG:
                            print(vars[i][1],
                                  f"added - 2 * {eq_inf} * {vars[i][0]} * {dict_con['rhs']} + {eq_inf} * {vars[i][0]} ** 2", "to",
                                  self.linear[vars[i][1]])
                        self.linear[vars[i][1]] = self.linear[vars[i][1]] - 2 * eq_inf * vars[i][0] * dict_con['rhs'] \
                                                  + eq_inf * vars[i][0] ** 2
                    else:

                        self.linear[vars[i][1]] = -2 * eq_inf * vars[i][0] * dict_con['rhs'] + eq_inf * vars[i][0] ** 2
                        if DEBUG:
                            print(vars[i][1], f"added -2 * {eq_inf} * {vars[i][0]} * {dict_con['rhs']} + {eq_inf} * {vars[i][0]} ** 2")
                            print("Adding a new linear term for placement", vars[i][1])

                # Terms involving pairwise products
                for i in range(N):
                    for j in range(i + 1, N):
                        pair = BQM.order(vars[i][1], vars[j][1])

                        if pair in self.quadratic:
                            if DEBUG:
                                print(pair, f"added 2 * {eq_inf} * {vars[i][0]} * {vars[j][0]}", "to", self.quadratic[pair])
                            self.quadratic[pair] = self.quadratic[pair] + 2 * eq_inf * vars[i][0] * vars[j][0]
                        else:
                            self.quadratic[pair] = 2 * eq_inf * vars[i][0] * vars[j][0]
                            if DEBUG:
                                print(pair, f"added 2 * {eq_inf} * {vars[i][0]} * {vars[j][0]}")
                                print("Adding new quadratic term for placement", pair)

            else:
                print("Encountered constraint of unknown type")

    @staticmethod
    def specie_pos(name):
        m = BQM.p_ion.search(name)
        return m.group('specie'), int(m.group('pos'))

    @staticmethod
    def order(t1, t2):
        '''
        tuples are of the form ('O', 2)
        order is based on position first, where lower is better
        and then on the species

        Return t1, t2 in the correct order
        '''

        swap = False

        if t1[1] == t2[1]:
            if t1[0] > t2[0]:
                swap = True

        if t1[1] > t2[1]:
            swap = True

        if swap:
            return t2, t1
        else:
            return t1, t2


if __name__ == "__main__":
    bqm_model = BQM()
    bqm_model.parse_lp("model.lp")
    bqm_model.parse_constraints()
    # print(bqm_model.constraints_dict)
    # print(bqm_model.quadratic)
    # for k in bqm_model.quadratic:
    #     if ('Y', 95) in k:
    #         print(k)
    bqm_model.qubofy(100, 300)
