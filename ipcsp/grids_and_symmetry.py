"""
Code generating lattices (grids) as well as dealing with symmetry constraints.
Space groups are stored in G{space group number}.txt files in filedir given
by their operations. CO{grid}G{group number}.json files contain information about the
positions that belong to the same orbit of a given group and must contain the same atoms.
"""
import numpy as np
import json
from ipcsp import root_dir

filedir = root_dir / '/data/grids/'


def generate_orthorhombic(ions_on_sides):
    """
    Ions_on_side=4 will generate points with x coordinates 0, 0.25, 0.5 and 0.75.
    Note that 1 is kind of another cell already. So, side/ions is the step size
    There will be prod(ions_on_side) points in total in the cell.
    Check separately whether you get charge NEUTRAL system!
    """

    step = 1.0 / np.array(ions_on_sides)

    pos = np.zeros((ions_on_sides[0] * ions_on_sides[1] * ions_on_sides[2], 3))
    # print("The total number of points in the cell is ", len(self.ions))

    row = 0
    for (i, j, k) in np.ndindex(ions_on_sides[0], ions_on_sides[1], ions_on_sides[2]):
        pos[row,] = np.array([i * step[0], j * step[1], k * step[2]])
        row = row + 1
    return pos


def cubic(ions_on_side):
    """
    generate points for the cubic grid
    """
    return generate_orthorhombic([ions_on_side, ions_on_side, ions_on_side])


def readGroup(number):
    '''
    Returns the list of augmented matrices of a space group with given number.
    '''
    result = []
    with open(filedir / 'G{number}.txt'.format(number=number)) as f:
        for line in f.readlines():
            line = line.rstrip('\n')
            line = line.split(sep=',')
            array = []
            for elem in line:
                if "/" in elem:
                    a, b = elem.split(sep='/')
                    # print(elem, ':', int(a)/int(b))
                    array.append(int(a) / int(b))
                else:
                    array.append(int(elem))
            # line = list(map(int, line))

            result.append(np.reshape(np.array(array), (4, 4)))
            # print(np.array(line))
            # print(np.reshape(np.array(array), (4,4)))
    return result


def generate_cubic_orbits(ions_on_side, group):
    transformations = readGroup(group)
    points = cubic(ions_on_side)

    orbits = {}
    remaining_points = [(len(points) - i - 1) for i in range(len(points))]

    while len(remaining_points) > 0:
        point = remaining_points.pop()
        orbits[point] = []

        for t in transformations:
            equiv_point = (t @ np.append(points[point], 1))[0:3]
            for i in range(3):
                if equiv_point[i] < -0.0001:
                    equiv_point[i] = 1 + equiv_point[i]

            for idx, p in enumerate(points):
                if (np.linalg.norm(p - equiv_point) < 0.0001):
                    if idx in remaining_points:
                        remaining_points.remove(idx)
                        orbits[point].append(idx)
                    break
    print(ions_on_side, ':', len(orbits))
    return orbits


if __name__ == "__main__":

    positions = cubic(16)
    with open('./data/grids/CO16G230.json', "r") as f:
        orbits = json.load(f)

    orb_key = list(orbits.keys())

    for o in orb_key:
        print(positions[int(o)])

    counts = {}
    for o in orb_key:
        size = len(orbits[o]) + 1
        if size in counts:
            counts[size] += 1
        else:
            counts[size] = 1
            # print(o, ':', len(orbits[o]) + 1, end=', ')

    print(counts.so)
    exit()
    # A piece to read orbit files
    # group = '215'
    # print("Group ", group)
    # for size in range(2, 4):
    #     with open(filedir+f'CO{size}G{group}.json', "r") as read_file:
    #         print(size, len(json.load(read_file)))
    # exit()

    # A piece to generate orbit files
    # group = '195'
    # group = '206'
    # group = '218'
    # group = '227'
    # group = '230'
    group = '215'
    for size in range(2, 13):
        with open(filedir + 'CO{size}G{group}.json'.format(size=size, group=group), "w") as write_file:
            json.dump(generate_cubic_orbits(size, group), write_file)
    exit()

    print(generate_cubic_orbits(6, '221'))
    exit()
    # size_range = range(2,17)
    # ouput_directory = filedir
    # for size in size_range:
    #     np.savetxt(filedir+f'C{size}.txt', cubic(size))
    # readGroup('221')
    matrices = readGroup('221')
    d = cubic(3)
    # m = np.array([ [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
    # m = np.array([ 0,-1,0,0, 1,0,0,0, 0,0,1,0, 0,0,0,1])
    # m = m.reshape((4,4))
    print(matrices[3])

    for pos in d[0:5]:
        print(np.append(pos, 1), '->', (matrices[3] @ np.append(pos, 1))[0:3])
        # print(d@m)
