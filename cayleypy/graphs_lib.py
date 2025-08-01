"""Library of pre-defined graphs."""

# pylint: disable=line-too-long

from itertools import permutations, combinations

from .cayley_graph_def import CayleyGraphDef, MatrixGenerator
from .permutation_utils import transposition, permutation_from_cycles
from .puzzles.hungarian_rings import get_santa_parameters_from_n
from .puzzles.puzzles import Puzzles


def _create_coxeter_generators(n: int) -> list[list[int]]:
    return [transposition(n, k, k + 1) for k in range(n - 1)]


class PermutationGroups:
    """Pre-defined Cayley graphs for permutation groups (S_n)."""

    @staticmethod
    def all_transpositions(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n>=2), generated by all n(n-1)/2 transpositions."""
        assert n >= 2
        generators = []
        generator_names = []
        for i in range(n):
            for j in range(i + 1, n):
                generators.append(transposition(n, i, j))
                generator_names.append(f"({i},{j})")
        return CayleyGraphDef.create(generators, central_state=list(range(n)), generator_names=generator_names)

    @staticmethod
    def full_reversals(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n>=2), generated by reverses of all possible n(n-1)/2 substrings."""
        assert n >= 2
        generators = []
        generator_names = []
        for i in range(n):
            for j in range(i + 1, n):
                perm = list(range(i)) + list(range(j, i - 1, -1)) + list(range(j + 1, n))
                generators.append(perm)
                generator_names.append(f"R[{i}..{j}]")
        return CayleyGraphDef.create(generators, central_state=list(range(n)), generator_names=generator_names)

    @staticmethod
    def signed_reversals(n: int) -> CayleyGraphDef:
        """Cayley graph generated by reverses of all possible signed n(n+1)/2 substrings.

        Actually is a graph for S_2n (n>=1) representing a graph for n elements, where i-th index represents bottom
        side of i-th element, and (n+i)-th index represents top side of i-th element. The graph has n generators
        denoted R[1..1],R[1..2]..R[n..n], where R[i..j] is signed reverse of elements i,i+1..j
        (or actually indexes i..j,n+i..n+j).
        """
        assert n >= 1
        generators = []
        generator_names = []
        for i in range(n):
            for j in range(i, n):
                perm = []
                perm += list(range(i))
                perm += list(range(n + j, n + i - 1, -1))
                perm += list(range(j + 1, n))
                perm += list(range(n, n + i))
                perm += list(range(j, i - 1, -1))
                perm += list(range(n + j + 1, n + n))
                generators.append(perm)
                generator_names.append(f"R[{i}..{j}]")
        return CayleyGraphDef.create(generators, central_state=list(range(2 * n)), generator_names=generator_names)

    @staticmethod
    def lrx(n: int, k: int = 1) -> CayleyGraphDef:
        """Cayley graph for S_n (n>=3), generated by: shift left, shift right, swap two elements 0 and k.

        :param n: Size of permutations.
        :param k: Specifies that X is transposition of elements 0 and k. 1<=k<n.
            By default, k=1, which means X is transposition of first 2 elements.
        """
        assert n >= 3
        generators = [list(range(1, n)) + [0], [n - 1] + list(range(0, n - 1)), transposition(n, 0, k)]
        generator_names = ["L", "R", "X"]
        name = f"lrx-{n}"
        if k != 1:
            name += f"(k={k})"
        return CayleyGraphDef.create(
            generators,
            central_state=list(range(n)),
            generator_names=generator_names,
            name=name,
        )

    @staticmethod
    def lx(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n>=3), generated by left shift (L) and swapping first two elements (X).

        This is an example of a Cayley graph where generators are not inverse-closed.

        See https://oeis.org/A039745.

        :param n: Size of permutations.
        """
        assert n >= 3
        generators = [list(range(1, n)) + [0], transposition(n, 0, 1)]
        generator_names = ["L", "X"]
        return CayleyGraphDef.create(
            generators, central_state=list(range(n)), generator_names=generator_names, name=f"lx-{n}"
        )

    @staticmethod
    def top_spin(n: int, k: int = 4):
        """Cayley graph for S_n (n>=k>=2), generated by: shift left, shift right, reverse first k elements.

        :param n: Size of permutations.
        :param k: Specifies size of prefix to reverse. By default, k=4.
        """
        assert n >= k >= 2
        generators = [
            list(range(1, n)) + [0],
            [n - 1] + list(range(0, n - 1)),
            list(range(k - 1, -1, -1)) + list(range(k, n)),
        ]
        name = f"top_spin-{n}-{k}"
        return CayleyGraphDef.create(generators, central_state=list(range(n)), name=name)

    @staticmethod
    def coxeter(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n>=2), generated by adjacent transpositions (Coxeter generators).

        It has n-1 generators: (0,1), (1,2), ..., (n-2,n-1).
        """
        assert n >= 2
        generators = _create_coxeter_generators(n)
        generator_names = [f"({i},{i + 1})" for i in range(n - 1)]
        central_state = list(range(n))
        name = f"coxeter-{n}"
        return CayleyGraphDef.create(
            generators, central_state=central_state, generator_names=generator_names, name=name
        )

    @staticmethod
    def cyclic_coxeter(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n>=2), generated by adjacent transpositions plus cyclic transposition.

        It has n generators: (0,1), (1,2), ..., (n-2,n-1), (0,n-1).
        """
        assert n >= 2
        generators = _create_coxeter_generators(n) + [transposition(n, 0, n - 1)]
        generator_names = [f"({i},{i + 1})" for i in range(n - 1)] + [f"(0,{n - 1})"]
        central_state = list(range(n))
        name = f"cyclic_coxeter-{n}"
        return CayleyGraphDef.create(
            generators, central_state=central_state, generator_names=generator_names, name=name
        )

    @staticmethod
    def pancake(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n>=2), generated by reverses of all prefixes.

        It has n-1 generators denoted R1,R2..R(n-1), where Ri is reverse of elements 0,1..i.
        See https://en.wikipedia.org/wiki/Pancake_graph.
        """
        assert n >= 2
        generators = []
        generator_names = []
        for prefix_len in range(2, n + 1):
            perm = list(range(prefix_len - 1, -1, -1)) + list(range(prefix_len, n))
            generators.append(perm)
            generator_names.append("R" + str(prefix_len - 1))
        name = f"pancake-{n}"
        return CayleyGraphDef.create(
            generators, central_state=list(range(n)), generator_names=generator_names, name=name
        )

    @staticmethod
    def cubic_pancake(n: int, subset: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n>=2), generated by set of 3 prefix reversal generators.

        Sets definitions are:
          - subset=1 => {Rn, R(n-1), R2}
          - subset=2 => {Rn, R(n-1), R3}
          - subset=3 => {Rn, R(n-1), R(n-2)}
          - subset=4 => {Rn, R(n-1), R(n-3)}
          - subset=5 => {Rn, R(n-2), R2}
          - subset=6 => {Rn, R(n-2), R3}
          - subset=7 => {Rn, R(n-2), R(n-3)}

        where Ri is reverse of elements 0,1..i.
        """

        def pancake_generator(k: int, n: int):
            return list(range(k - 1, -1, -1)) + list(range(k, n, 1))

        assert n >= 2
        assert subset in [1, 2, 3, 4, 5, 6, 7], "subset parameter must be one of {1,2,3,4,5,6,7}"
        generators = []
        generator_names = []
        if subset == 1:
            generators = [pancake_generator(n, n), pancake_generator(n - 1, n), pancake_generator(2, n)]
            generator_names = [f"R{n}", f"R{n-1}", "R2"]
        elif subset == 2:
            generators = [pancake_generator(n, n), pancake_generator(n - 1, n), pancake_generator(3, n)]
            generator_names = [f"R{n}", f"R{n-1}", "R3"]
        elif subset == 3:
            generators = [pancake_generator(n, n), pancake_generator(n - 1, n), pancake_generator(n - 2, n)]
            generator_names = [f"R{n}", f"R{n-1}", f"R{n-2}"]
        elif subset == 4:
            generators = [pancake_generator(n, n), pancake_generator(n - 1, n), pancake_generator(n - 3, n)]
            generator_names = [f"R{n}", f"R{n-1}", f"R{n-3}"]
        elif subset == 5:
            generators = [pancake_generator(n, n), pancake_generator(n - 2, n), pancake_generator(2, n)]
            generator_names = [f"R{n}", f"R{n-2}", "R2"]
        elif subset == 6:
            generators = [pancake_generator(n, n), pancake_generator(n - 2, n), pancake_generator(3, n)]
            generator_names = [f"R{n}", f"R{n-2}", "R3"]
        elif subset == 7:
            generators = [pancake_generator(n, n), pancake_generator(n - 2, n), pancake_generator(n - 3, n)]
            generator_names = [f"R{n}", f"R{n-2}", f"R{n-3}"]
        name = f"cubic_pancake-{n}-{subset}"
        return CayleyGraphDef.create(
            generators, central_state=list(range(n)), generator_names=generator_names, name=name
        )

    @staticmethod
    def burnt_pancake(n: int) -> CayleyGraphDef:
        """Cayley graph generated by reverses of all signed prefixes.

        Actually is a graph for S_2n (n>=1) representing a graph for n pancakes, where i-th element represents bottom
        side of i-th pancake, and (n+i)-th element represents top side of i-th pancake. The graph has n generators
        denoted R1,R2..R(n), where Ri is reverse of elements 0,1..i,n,n+1..n+i."""
        assert n >= 1
        generators = []
        generator_names = []
        for prefix_len in range(0, n):
            perm = []
            perm += list(range(n + prefix_len, n - 1, -1))
            perm += list(range(prefix_len + 1, n, 1))
            perm += list(range(prefix_len, -1, -1))
            perm += list(range(n + prefix_len + 1, 2 * n, 1))
            generators.append(perm)
            generator_names.append("R" + str(prefix_len + 1))
        name = f"burnt_pancake-{n}"
        return CayleyGraphDef.create(
            generators, central_state=list(range(2 * n)), generator_names=generator_names, name=name
        )

    @staticmethod
    def three_cycles(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n ≥ 3), generated by all 3-cycles (a, b, c) where a < b, a < c."""
        assert n >= 3
        generators = []
        generator_names = []
        for a, b, c in permutations(range(n), 3):
            if a < b and a < c:
                generators.append(permutation_from_cycles(n, [[a, b, c]]))
                generator_names.append(f"({a} {b} {c})")
        name = f"three_cycles-{n}"
        return CayleyGraphDef.create(
            generators, central_state=list(range(n)), generator_names=generator_names, name=name
        )

    @staticmethod
    def three_cycles_0ij(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n ≥ 3), generated by 3-cycles of the form (0 i j), where i != j."""
        generators = []
        generator_names = []
        for i, j in permutations(range(1, n), 2):
            generators.append(permutation_from_cycles(n, [[0, i, j]]))
            generator_names.append(f"({0} {i} {j})")
        name = f"three_cycles_0ij-{n}"
        return CayleyGraphDef.create(
            generators, central_state=list(range(n)), generator_names=generator_names, name=name
        )

    @staticmethod
    def derangements(n: int) -> CayleyGraphDef:
        """Cayley graph generated by permutations without fixed points, called derangements."""
        assert n >= 2
        generators = []
        generator_names = []
        for idx, perm in enumerate(permutations(range(n))):
            has_fixed_point = any(perm[i] == i for i in range(n))
            if not has_fixed_point:
                generators.append(list(perm))
                generator_names.append(f"D{idx}")
        name = f"derangements-{n}"
        return CayleyGraphDef.create(
            generators, central_state=list(range(n)), generator_names=generator_names, name=name
        )

    @staticmethod
    def stars(n: int) -> CayleyGraphDef:
        """Cayley graph generated by stars permutations."""
        assert n >= 3
        generators = []
        generator_names = []
        for i in range(1, n):
            generators.append(transposition(n, 0, i))
            generator_names.append(f"S{i}")
        name = f"stars-{n}"
        return CayleyGraphDef.create(
            generators, central_state=list(range(n)), generator_names=generator_names, name=name
        )

    @staticmethod
    def rapaport_m1(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n with M1 generators.

        Reference: E. Rapaport-Strasser. Cayley color groups and hamilton lines. Scr. Math, 24:51–58, 1959.
        """
        generators = []
        generator_names = []

        # Generator 1: Transpositions: (0,1), (0,1)(2,3), (0,1)(2,3)(4,5),...
        for num_pairs in range(1, (n // 2) + 1):
            cycles = []
            for idx in range(num_pairs):
                if 2 * idx + 1 < n:
                    cycles.append([2 * idx, 2 * idx + 1])
            permutation = permutation_from_cycles(n, cycles)
            generators.append(permutation)
            generator_names.append(f"M1_0_{num_pairs}")

        # Generator 2: Transpositions: (1,2), (1,2)(3,4), (1,2)(3,4)(5,6),...
        for num_pairs in range(1, ((n - 1) // 2) + 1):
            cycles = []
            permutation = list(range(n))
            for idx in range(num_pairs):
                if 1 + 2 * idx + 1 < n:
                    cycles.append([1 + 2 * idx, 1 + 2 * idx + 1])
            permutation = permutation_from_cycles(n, cycles)
            generators.append(permutation)
            generator_names.append(f"M1_1_{num_pairs}")

        name = f"rapaport_m1-{n}"
        return CayleyGraphDef.create(
            generators, central_state=list(range(n)), generator_names=generator_names, name=name
        )

    @staticmethod
    def rapaport_m2(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n with M2 generators.

        Reference: E. Rapaport-Strasser. Cayley color groups and hamilton lines. Scr. Math, 24:51–58, 1959.
        """
        # Generator 1: Transposition (0,1)
        g1 = transposition(n, 0, 1)

        # Generator 2: Product of transpositions (0,1)(2,3)...
        g2 = list(range(n))
        for i in range(0, n - 1, 2):
            g2[i], g2[i + 1] = g2[i + 1], g2[i]

        # Generator 3: Product of transpositions (1,2)(3,4)...
        g3 = list(range(n))
        for i in range(1, n - 1, 2):
            g3[i], g3[i + 1] = g3[i + 1], g3[i]

        generators = [g1, g2, g3]
        generator_names = ["(0,1)", "EvenDisjTrans", "OddDisjTrans"]

        name = f"rapaport_m2-{n}"
        return CayleyGraphDef.create(
            generators, central_state=list(range(n)), generator_names=generator_names, name=name
        )

    @staticmethod
    def all_cycles(n: int) -> CayleyGraphDef:
        """Cayley graph for S_n (n ≥ 2), generated by all cycles of length 2 to n."""
        assert n >= 2
        generators = []
        generator_names = []

        for k in range(2, n + 1):
            for subset in combinations(range(n), k):
                min_elem = min(subset)
                rest = [x for x in subset if x != min_elem]

                for perm in permutations(rest):
                    cycle = list(range(n))
                    current = min_elem
                    for target in perm:
                        cycle[current] = target
                        current = target
                    cycle[current] = min_elem

                    generators.append(cycle)
                    generator_names.append(f"cycle_{len(generators)}")

        name = f"all_cycles-{n}"
        return CayleyGraphDef.create(
            generators, central_state=list(range(n)), generator_names=generator_names, name=name
        )

    @staticmethod
    def wrapped_k_cycles(n: int, k: int) -> CayleyGraphDef:
        """
        Cayley graph for S_n (n >= 2, 2 <= k <= n), generated by all consecutive k-cycles with wrap-around.
        """
        assert n >= 2 and 2 <= k <= n, "Need n >= 2 and 2 <= k <= n"
        generators = []
        generator_names = []
        for start in range(n):
            cycle = [(start + j) % n for j in range(k)]
            generators.append(permutation_from_cycles(n, [cycle]))
            generator_names.append(f"({' '.join(map(str, cycle))})")
        name = f"wrapped_k_cycles-{n}-{k}"
        return CayleyGraphDef.create(
            generators, central_state=list(range(n)), generator_names=generator_names, name=name
        )


def prepare_graph(name: str, n: int = 0, **unused_kwargs) -> CayleyGraphDef:
    """Returns pre-defined CayleyGraphDef by codename and additional kwargs.

    See the source of this function for list of supported graphs.
    """
    if name == "cube_2/2/2_6gensQTM":
        return Puzzles.rubik_cube(2, "fixed_QTM")
    elif name == "cube_2/2/2_9gensHTM":
        return Puzzles.rubik_cube(2, "fixed_HTM")
    elif name == "cube_3/3/3_12gensQTM":
        return Puzzles.rubik_cube(3, "QTM")
    elif name == "cube_3/3/3_18gensHTM":
        return Puzzles.rubik_cube(3, "HTM")
    elif name == "mini_pyramorphix":
        return Puzzles.mini_pyramorphix()
    elif name == "pyraminx":
        return Puzzles.pyraminx()
    elif name == "hungarian_rings":
        hr_params = get_santa_parameters_from_n(n)
        return Puzzles.hungarian_rings(*hr_params)
    elif name == "starminx":
        return Puzzles.starminx()
    elif name == "starminx_2":
        return Puzzles.starminx_2()
    elif name == "megaminx":
        return Puzzles.megaminx()
    elif name == "lx":
        return PermutationGroups.lx(n)
    elif name.startswith("lx-"):
        return PermutationGroups.lrx(int(name[3:]))
    elif name == "lrx":
        return PermutationGroups.lrx(n)
    elif name.startswith("lrx-"):
        return PermutationGroups.lrx(int(name[4:]))
    elif name == "top_spin":
        return PermutationGroups.top_spin(n)
    elif name == "all_transpositions":
        return PermutationGroups.all_transpositions(n)
    elif name == "full_reversals":
        return PermutationGroups.full_reversals(n)
    elif name == "coxeter":
        return PermutationGroups.coxeter(n)
    elif name == "pancake":
        return PermutationGroups.pancake(n)
    elif name == "all_cycles":
        return PermutationGroups.all_cycles(n)
    else:
        raise ValueError(f"Unknown generator set: {name}")


class MatrixGroups:
    """Pre-defined Cayley graphs for matrix groups."""

    @staticmethod
    def heisenberg(modulo: int = 0) -> CayleyGraphDef:
        """Returns Cayley graph for the Heisenberg group.

        This is a group of upper triangular 3x3 integer matrices with 1s on main diagonal.
        See https://en.wikipedia.org/wiki/Heisenberg_group.

        Generated by 4 matrices: x=(110,010,001), y=(100,011,001), and their inverses.
        Central element is identity matrix.

        :param modulo: multiplication modulo (or 0 if multiplication is not modular).
        :return: requested graph as `CayleyGraphDef`.
        """
        x = MatrixGenerator.create([[1, 1, 0], [0, 1, 0], [0, 0, 1]], modulo=modulo)
        y = MatrixGenerator.create([[1, 0, 0], [0, 1, 1], [0, 0, 1]], modulo=modulo)
        name = "heisenberg"
        if modulo > 0:
            name += f"%{modulo}"
        return CayleyGraphDef.for_matrix_group(
            generators=[x, x.inv, y, y.inv],
            generator_names=["x", "x'", "y", "y'"],
            name=name,
        )
