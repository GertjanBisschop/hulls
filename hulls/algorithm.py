import bintrees
import copy
import math
import msprime
import numpy as np
import random
import sys
import tskit

import hulls.verify as verify


class FenwickTree:
    """
    A Fenwick Tree to represent cumulative frequency tables over
    integers. Each index from 1 to max_index initially has a
    zero frequency.

    This is an implementation of the Fenwick tree (also known as a Binary
    Indexed Tree) based on "A new data structure for cumulative frequency
    tables", Software Practice and Experience, Vol 24, No 3, pp 327 336 Mar
    1994. This implementation supports any non-negative frequencies, and the
    search procedure always returns the smallest index such that its cumulative
    frequency <= f. This search procedure is a slightly modified version of
    that presented in Tech Report 110, "A new data structure for cumulative
    frequency tables: an improved frequency-to-symbol algorithm." available at
    https://www.cs.auckland.ac.nz/~peter-f/FTPfiles/TechRep110.ps
    """

    def __init__(self, max_index):
        assert max_index > 0
        self.__max_index = max_index
        self.__tree = [0 for j in range(max_index + 1)]
        self.__value = [0 for j in range(max_index + 1)]
        # Compute the binary logarithm of max_index
        u = self.__max_index
        while u != 0:
            self.__log_max_index = u
            u -= u & -u

    def get_total(self):
        """
        Returns the total cumulative frequency over all indexes.
        """
        return self.get_cumulative_sum(self.__max_index)

    def increment(self, index, v):
        """
        Increments the frequency of the specified index by the specified
        value.
        """
        assert 0 < index <= self.__max_index
        self.__value[index] += v
        j = index
        while j <= self.__max_index:
            self.__tree[j] += v
            j += j & -j

    def set_value(self, index, v):
        """
        Sets the frequency at the specified index to the specified value.
        """
        f = self.get_value(index)
        self.increment(index, v - f)

    def get_cumulative_sum(self, index):
        """
        Returns the cumulative frequency of the specified index.
        """
        assert 0 < index <= self.__max_index
        j = index
        s = 0
        while j > 0:
            s += self.__tree[j]
            j -= j & -j
        return s

    def get_value(self, index):
        """
        Returns the frequency of the specified index.
        """
        return self.__value[index]

    def find(self, v):
        """
        Returns the smallest index with cumulative sum >= v.
        """
        j = 0
        s = v
        half = self.__log_max_index
        while half > 0:
            # Skip non-existant entries
            while j + half > self.__max_index:
                half >>= 1
            k = j + half
            if s > self.__tree[k]:
                j = k
                s -= self.__tree[j]
            half >>= 1
        return j + 1


class Segment:
    """
    A class representing a single segment. Each segment has a left
    and right, denoting the loci over which it spans, a node and a
    next, giving the next in the chain.
    """

    def __init__(self, index):
        self.left = None
        self.right = None
        self.node = None
        self.prev = None
        self.next = None
        self.population = None
        self.label = 0
        self.index = index
        self.hull = None

    def __repr__(self):
        return repr((self.left, self.right, self.node, self.index))

    @staticmethod
    def show_chain(seg):
        s = ""
        while seg is not None:
            s += f"[{seg.left}, {seg.right}: {seg.node}], "
            seg = seg.next
        return s[:-2]

    def __lt__(self, other):
        return (self.left, self.right, self.population, self.node) < (
            other.left,
            other.right,
            other.population,
            self.node,
        )

    def get_left_end(self):
        seg = self
        while seg is not None:
            left = seg.left
            seg = seg.prev

        return left

    def get_hull(self):
        seg = self
        while seg is not None:
            hull = seg.hull
            seg = seg.prev

        return hull

    def get_right_end(self):
        seg = self
        while seg is not None:
            right = seg.right
            seg = seg.next

        return right

    def get_left_index(self):
        seg = self
        while seg is not None:
            index = seg.index
            seg = seg.prev

        return index


class Population:
    """
    Class representing a population in the simulation.
    """

    def __init__(self, id_, num_labels=1, max_segments=100):
        self.id = id_
        self.start_time = 0
        self.start_size = 1.0
        self.growth_rate = 0
        # Keep a list of each label.
        # We'd like to use AVLTrees here for P but the API doesn't quite
        # do what we need. Lists are inefficient here and should not be
        # used in a real implementation.
        self._ancestors = [[] for _ in range(num_labels)]

        # ADDITIONAL STATES FOR SMC(k)
        # this has to be done for each label
        # track hulls based on left
        self.hulls_left = [OrderStatisticsTree() for _ in range(num_labels)]
        self.coal_mass_index = [FenwickTree(max_segments) for j in range(num_labels)]
        # track rank of hulls right
        self.hulls_right = [OrderStatisticsTree() for _ in range(num_labels)]

    def print_state(self):
        print("Population ", self.id)
        print("\tstart_size = ", self.start_size)
        print("\tgrowth_rate = ", self.growth_rate)
        print("\tAncestors: ", len(self._ancestors))
        for label, ancestors in enumerate(self._ancestors):
            print("\tLabel = ", label)
            for u in ancestors:
                print("\t\t" + Segment.show_chain(u))
            print(self.hulls_left)

    def set_growth_rate(self, growth_rate, time):
        # TODO This doesn't work because we need to know what the time
        # is so we can set the start size accordingly. Need to look at
        # ms's model carefully to see what it actually does here.
        new_size = self.get_size(time)
        self.start_size = new_size
        self.start_time = time
        self.growth_rate = growth_rate

    def set_start_size(self, start_size):
        self.start_size = start_size
        self.growth_rate = 0

    def get_num_ancestors(self, label=None):
        if label is None:
            return sum(len(label_ancestors) for label_ancestors in self._ancestors)
        else:
            return len(self._ancestors[label])

    def get_num_pairs(self, label=None):
        # can be improved by updating values in self.num_pairs
        if label is None:
            return sum(mass_index.get_total() for mass_index in self.coal_mass_index)
        else:
            return self.coal_mass_index[label].get_total()

    def get_size(self, t):
        """
        Returns the size of this population at time t.
        """
        dt = t - self.start_time
        return self.start_size * math.exp(-self.growth_rate * dt)

    def get_common_ancestor_waiting_time(self, t, rng=None):
        """
        Returns the random waiting time until a common ancestor event
        occurs within this population.
        """
        if rng is None:
            rng = random.Random()
        ret = sys.float_info.max
        k = self.get_num_pairs()
        if k > 0:
            u = rng.expovariate(k / 2)  # divide by 2???
            if self.growth_rate == 0:
                ret = self.start_size * u
            else:
                dt = t - self.start_time
                z = (
                    1
                    + self.growth_rate
                    * self.start_size
                    * math.exp(-self.growth_rate * dt)
                    * u
                )
                if z > 0:
                    ret = math.log(z) / self.growth_rate
        return ret

    def get_ind_range(self, t):
        """Returns ind labels at time t"""
        first_ind = np.sum([self.get_size(t_prev) for t_prev in range(0, int(t))])
        last_ind = first_ind + self.get_size(t)

        return range(int(first_ind), int(last_ind) + 1)

    def increment_avl(self, ost, coal_mass, hull, increment):
        right = hull.right
        curr_hull = hull
        curr_hull, _ = ost.succ_key(curr_hull)
        while curr_hull is not None:
            if right > curr_hull.left:
                ost.avl[curr_hull] += increment
                coal_mass.increment(curr_hull.index, increment)
            else:
                break
            curr_hull, _ = ost.succ_key(curr_hull)

    def reset_hull_right(self, label, hull, old_right, new_right):
        # when resetting the hull.right of a pre-existing hull we need to
        # decrement count of all lineages starting off between hull.left and bp
        # FIX: logic is almost identical as increment_avl()!!!
        ost = self.hulls_left[label]
        curr_hull = hull
        curr_hull, _ = ost.succ_key(curr_hull)
        while curr_hull is not None:
            if curr_hull.left >= old_right:
                break
            if curr_hull.left >= new_right:
                ost.avl[curr_hull] -= 1
                self.coal_mass_index[label].increment(curr_hull.index, -1)
            curr_hull, _ = ost.succ_key(curr_hull)
        hull.right = new_right

        # adjust rank of hull.right
        ost = self.hulls_right[label]
        floor = ost.floor_key(HullEnd(old_right))
        assert floor.x == old_right
        ost.pop(floor)
        insertion_order = 0
        hull_end = HullEnd(new_right)
        floor = ost.floor_key(hull_end)
        if floor is not None:
            if floor.x == hull_end.x:
                insertion_order = floor.insertion_order + 1
        hull_end.insertion_order = insertion_order
        ost[hull_end] = 0

    def remove_hull(self, label, hull):
        ost = self.hulls_left[label]
        coal_mass_index = self.coal_mass_index[label]
        self.increment_avl(ost, coal_mass_index, hull, -1)
        # adjust insertion order
        curr_hull, _ = ost.succ_key(hull)
        count, left_rank = ost.pop(hull)
        while curr_hull is not None:
            if curr_hull.left == hull.left:
                curr_hull.insertion_order -= 1
            else:
                break
            curr_hull, _ = ost.succ_key(curr_hull)
        ost = self.hulls_right[label]
        floor = ost.floor_key(HullEnd(hull.right))
        assert floor.x == hull.right
        _, right_rank = ost.pop(floor)
        hull.insertion_order = math.inf
        self.coal_mass_index[label].set_value(hull.index, 0)

    def remove(self, individual, label=0, hull=None):
        """
        Removes and returns the individual at the specified index.
        """
        # update hull information
        assert individual.left == individual.get_left_end()
        found = individual.get_hull()
        assert found == hull
        if hull is not None:
            self.remove_hull(label, hull)
        return self._ancestors[label].remove(individual)

    def remove_individual(self, individual, label=0):
        """
        Removes the given individual from its population.
        """
        return self._ancestors[label].remove(individual)

    def add_hull(self, label, hull):
        # logic left end
        ost_left = self.hulls_left[label]
        ost_right = self.hulls_right[label]
        insertion_order = 0
        num_starting_after_left = 0
        num_ending_before_left = 0

        floor = ost_left.floor_key(hull)
        if floor is not None:
            if floor.left == hull.left:
                insertion_order = floor.insertion_order + 1
            num_starting_after_left = ost_left.get_rank(floor) + 1
        hull.insertion_order = insertion_order

        floor = ost_right.floor_key(HullEnd(hull.left))
        if floor is not None:
            num_ending_before_left = ost_right.get_rank(floor) + 1
        count = num_starting_after_left - num_ending_before_left
        ost_left[hull] = count
        self.coal_mass_index[label].set_value(hull.index, count)

        # logic right end
        insertion_order = 0
        hull_end = HullEnd(hull.right)
        floor = ost_right.floor_key(hull_end)
        if floor is not None:
            if floor.x == hull.right:
                insertion_order = floor.insertion_order + 1
        hull_end.insertion_order = insertion_order
        ost_right[hull_end] = 0
        # self.num_pairs[label] += count - correction
        # Adjust counts for existing hulls in the avl tree
        coal_mass_index = self.coal_mass_index[label]
        self.increment_avl(ost_left, coal_mass_index, hull, 1)

    def add(self, individual, label=0, hull=None):
        """
        Inserts the specified individual into this population.
        """
        # update hull information
        if hull is not None:
            self.add_hull(label, hull)
        assert individual.label == label
        self._ancestors[label].append(individual)

    def __iter__(self):
        # will default to label 0
        # inter_label() extends behavior
        return iter(self._ancestors[0])

    def iter_label(self, label):
        """
        Iterates ancestors in popn from a label
        """
        return iter(self._ancestors[label])

    def iter_ancestors(self):
        """
        Iterates over all ancestors in a population over all labels.
        """
        for ancestors in self._ancestors:
            yield from ancestors

    def find_indv(self, indv):
        """
        find the index of an ancestor in population
        """
        return self._ancestors[indv.label].index(indv)


class RateMap:
    def __init__(self, positions, rates):
        self.positions = positions
        self.rates = rates
        self.cumulative = RateMap.recomb_mass(positions, rates)

    @staticmethod
    def recomb_mass(positions, rates):
        recomb_mass = 0
        cumulative = [recomb_mass]
        for i in range(1, len(positions)):
            recomb_mass += (positions[i] - positions[i - 1]) * rates[i - 1]
            cumulative.append(recomb_mass)
        return cumulative

    @property
    def sequence_length(self):
        return self.positions[-1]

    @property
    def total_mass(self):
        return self.cumulative[-1]

    @property
    def mean_rate(self):
        return self.total_mass / self.sequence_length

    def mass_between(self, left, right):
        left_mass = self.position_to_mass(left)
        right_mass = self.position_to_mass(right)
        return right_mass - left_mass

    def position_to_mass(self, pos):
        if pos == self.positions[0]:
            return 0
        if pos >= self.positions[-1]:
            return self.cumulative[-1]

        index = self._search(self.positions, pos)
        assert index > 0
        index -= 1
        offset = pos - self.positions[index]
        return self.cumulative[index] + offset * self.rates[index]

    def mass_to_position(self, recomb_mass):
        if recomb_mass == 0:
            return 0
        index = self._search(self.cumulative, recomb_mass)
        assert index > 0
        index -= 1
        mass_in_interval = recomb_mass - self.cumulative[index]
        pos = self.positions[index] + (mass_in_interval / self.rates[index])
        return pos

    def shift_by_mass(self, pos, mass):
        result_mass = self.position_to_mass(pos) + mass
        return self.mass_to_position(result_mass)

    def _search(self, values, query):
        left = 0
        right = len(values) - 1
        while left < right:
            m = (left + right) // 2
            if values[m] < query:
                left = m + 1
            else:
                right = m
        return left


class Hull:
    def __init__(self, index):
        self.left = None
        self.right = None
        self.ancestor_node = None
        self.index = index
        self.insertion_order = math.inf

    def __lt__(self, other):
        return (self.left, self.insertion_order) < (other.left, other.insertion_order)

    def __repr__(self):
        return f"l:{self.left}, r:{self.right}, io:{self.insertion_order}"


class HullEnd:
    def __init__(self, x):
        self.x = x
        self.insertion_order = math.inf

    def __lt__(self, other):
        return (self.x, self.insertion_order) < (other.x, other.insertion_order)

    def __repr__(self):
        return f"x:{self.x}, io:{self.insertion_order}"


class OrderStatisticsTree:
    def __init__(self):
        self.avl = bintrees.AVLTree()
        self.rank = {}
        self.size = 0
        self.min = None

    def __len__(self):
        return self.size

    def __setitem__(self, key, value):
        first = True
        rank = 0
        if self.min is not None:
            if self.min < key:
                prev_key = self.avl.floor_key(key)
                rank = self.rank[prev_key]
                rank += 1
                first = False
        if first:
            self.min = key
        self.avl[key] = value
        self.rank[key] = rank
        self.size += 1
        self.update_ranks(key, rank)

    def __getitem__(self, key):
        return self.avl[key], self.rank[key]

    def get_rank(self, key):
        return self.rank[key]

    def update_ranks(self, key, rank, increment=1):
        while rank < self.size - 1:
            key = self.avl.succ_key(key)
            self.rank[key] += increment
            rank += 1

    def pop(self, key):
        if self.min == key:
            if len(self) == 1:
                self.min = None
            else:
                self.min = self.avl.succ_key(key)
        rank = self.rank.pop(key)
        self.update_ranks(key, rank, -1)
        value = self.avl.pop(key)
        self.size -= 1
        return value, rank

    def succ_key(self, key):
        rank = self.rank[key]
        if rank < self.size - 1:
            key = self.avl.succ_key(key)
            rank += 1
            return key, rank
        else:
            return None, None

    def prev_key(self, key):
        if key == self.min:
            return None, None
        else:
            key = self.avl.prev_key(key)
            rank = self.rank[key]
            return key, rank

    def floor_key(self, key):
        if len(self) == 0:
            return None
        if key < self.min:
            return None
        return self.avl.floor_key(key)

    def ceil_key(self, key):
        if len(self) == 0:
            return None
        return self.avl.ceiling_key(key)


class Simulator:
    def __init__(
        self,
        *,
        initial_state=None,
        migration_matrix=None,
        max_segments=100,
        recombination_rate=0.0,
        gene_conversion_rate=0.0,
        gene_conversion_length=1,
        stop_condition=None,
        hull_offset=0,
        coalescing_segments_only=True,
        additional_nodes=None,
        random_seed=None,
        discrete_genome=True,
    ):
        if migration_matrix is None:
            migration_matrix = np.zeros((1, 1))
        N = migration_matrix.shape[0]
        assert len(initial_state.populations) == N
        for j in range(N):
            assert N == len(migration_matrix[j])
            assert migration_matrix[j][j] == 0
        self.migration_matrix = migration_matrix
        assert gene_conversion_length >= 1
        if migration_matrix is None:
            migration_matrix = np.zeros((N, N))
        self.migration_matrix = migration_matrix
        self.num_labels = 1
        self.num_populations = N
        population_sizes = [10_000 for _ in range(self.num_populations)]
        population_growth_rates = [0 for _ in range(self.num_populations)]
        if initial_state is None:
            raise ValueError("Requires an initial state.")
        if isinstance(initial_state, tskit.TreeSequence):
            initial_state = initial_state.dump_tables()
        self.tables = initial_state
        self.L = initial_state.sequence_length
        self.rng = random.Random(random_seed)

        self.recomb_map = RateMap([0, self.L], [recombination_rate, 0])
        self.gc_map = RateMap([0, self.L], [gene_conversion_rate, 0])
        self.tract_length = gene_conversion_length
        self.discrete_genome = discrete_genome

        self.max_segments = max_segments
        self.max_hulls = max_segments
        self.hull_offset = hull_offset
        self.segment_stack = []
        self.segments = [None for _ in range(self.max_segments + 1)]
        for j in range(self.max_segments):
            s = Segment(j + 1)
            self.segments[j + 1] = s
            self.segment_stack.append(s)
        # double check whether this is maintained/used.
        self.hull_stack = []
        self.hulls = [None for _ in range(self.max_hulls + 1)]
        for j in range(self.max_segments):
            h = Hull(j + 1)
            self.hulls[j + 1] = h
            self.hull_stack.append(h)
        self.S = bintrees.AVLTree()
        self.P = [
            Population(id_, self.num_labels, self.max_segments) for id_ in range(N)
        ]
        if self.recomb_map.total_mass == 0:
            self.recomb_mass_index = None
        else:
            self.recomb_mass_index = [
                FenwickTree(self.max_segments) for j in range(self.num_labels)
            ]
        if self.gc_map.total_mass == 0:
            self.gc_mass_index = None
        else:
            self.gc_mass_index = [
                FenwickTree(self.max_segments) for j in range(self.num_labels)
            ]
        self.S = bintrees.AVLTree()
        for pop in self.P:
            pop.set_start_size(population_sizes[pop.id])
            pop.set_growth_rate(population_growth_rates[pop.id], 0)
        self.edge_buffer = []
        self.modifier_events = [(sys.float_info.max, None, None)]

        self.initialise(initial_state)

        self.num_ca_events = 0
        self.num_re_events = 0
        self.num_gc_events = 0

        self.stop_condition = stop_condition
        self.coalescing_segments_only = coalescing_segments_only
        if additional_nodes is None:
            additional_nodes = 0
        self.additional_nodes = msprime.NodeType(additional_nodes)

    def initialise(self, initial_state):
        tables = initial_state
        ts = tables.tree_sequence()
        root_time = np.max(tables.nodes.time)
        self.t = root_time

        root_segments_head = [None for _ in range(ts.num_nodes)]
        root_segments_tail = [None for _ in range(ts.num_nodes)]
        last_S = -1
        for tree in ts.trees():
            left, right = tree.interval
            S = 0 if tree.num_roots == 1 else tree.num_roots
            if S != last_S:
                self.S[left] = S
                last_S = S
            # If we have 1 root this is a special case and we don't add in
            # any ancestral segments to the state.
            if tree.num_roots > 1:
                for root in tree.roots:
                    population = ts.node(root).population
                    if root_segments_head[root] is None:
                        seg = self.alloc_segment(left, right, root, population)
                        root_segments_head[root] = seg
                        root_segments_tail[root] = seg
                    else:
                        tail = root_segments_tail[root]
                        if tail.right == left:
                            tail.right = right
                        else:
                            seg = self.alloc_segment(
                                left, right, root, population, tail
                            )
                            tail.next = seg
                            root_segments_tail[root] = seg
        self.S[self.L] = -1

        # Insert the segment chains into the algorithm state.
        for node in range(ts.num_nodes):
            seg = root_segments_head[node]
            if seg is not None:
                left_end = seg.left
                pop = seg.population
                label = seg.label
                ancestor_node = seg
                self.P[seg.population].add(seg)
                while seg is not None:
                    self.set_segment_mass(seg)
                    right_end = seg.right
                    seg = seg.next
                new_hull = self.alloc_hull(left_end, right_end, ancestor_node)
                # insert Hull
                floor = self.P[pop].hulls_left[label].floor_key(new_hull)
                insertion_order = 0
                if floor is not None:
                    if floor.left == new_hull.left:
                        insertion_order = floor.insertion_order + 1
                new_hull.insertion_order = insertion_order
                self.P[pop].hulls_left[label][new_hull] = -1

        # initialise the correct coalesceable pairs count
        for pop in self.P:
            for label, ost_left in enumerate(pop.hulls_left):
                avl = ost_left.avl
                ost_right = pop.hulls_right[label]
                count = 0
                for hull in avl.keys():
                    floor = ost_right.floor_key(HullEnd(hull.left))
                    num_ending_before_hull = 0
                    if floor is not None:
                        num_ending_before_hull = ost_right.rank[floor] + 1
                    num_pairs = count - num_ending_before_hull
                    avl[hull] = num_pairs
                    pop.coal_mass_index[label].set_value(hull.index, num_pairs)
                    # insert HullEnd
                    hull_end = HullEnd(hull.right)
                    floor = ost_right.floor_key(hull_end)
                    insertion_order = 0
                    if floor is not None:
                        if floor.x == hull.right:
                            insertion_order = floor.insertion_order + 1
                    hull_end.insertion_order = insertion_order
                    ost_right[hull_end] = -1
                    count += 1

    def get_num_ancestors(self):
        return sum(pop.get_num_ancestors() for pop in self.P)

    def ancestors_remain(self):
        """
        Returns True if the simulation is not finished, i.e., there is some ancestral
        material that has not fully coalesced.
        """
        return self.get_num_ancestors() != 0

    def assert_stop_condition(self):
        """
        Returns true if the simulation is not finished given the global
        stopping condition that was specified.
        """
        if self.stop_condition is None:
            return self.ancestors_remain()
        elif self.stop_condition == "time":
            return True
        else:
            print("Error: unknown stop condition-", self.stop_condition)
            raise ValueError

    def change_population_size(self, pop_id, size):
        self.P[pop_id].set_start_size(size)

    def change_population_growth_rate(self, pop_id, rate, time):
        self.P[pop_id].set_growth_rate(rate, time)

    def change_migration_matrix_element(self, pop_i, pop_j, rate):
        self.migration_matrix[pop_i][pop_j] = rate

    def alloc_hull(self, left, right, ancestor_node):
        alpha = ancestor_node
        # while alpha.prev is not None:
        #    alpha = alpha.prev
        # assert alpha.left == left
        hull = self.hull_stack.pop()
        hull.left = left
        hull.right = min(right + self.hull_offset, self.L)
        hull.ancestor_node = alpha
        # assert alpha.prev is None
        while alpha is not None:
            alpha.hull = hull
            alpha = alpha.next
        return hull

    def alloc_segment(
        self,
        left,
        right,
        node,
        population,
        prev=None,
        next=None,  # noqa: A002
        label=0,
        hull=None,
    ):
        """
        Pops a new segment off the stack and sets its properties.
        """
        s = self.segment_stack.pop()
        s.left = left
        s.right = right
        s.node = node
        s.population = population
        s.next = next
        s.prev = prev
        s.label = label
        s.hull = hull
        return s

    def copy_segment(self, segment):
        return self.alloc_segment(
            left=segment.left,
            right=segment.right,
            node=segment.node,
            population=segment.population,
            next=segment.next,
            prev=segment.prev,
            label=segment.label,
            hull=segment.hull,
        )

    def free_segment(self, u):
        """
        Frees the specified segment making it ready for reuse and
        setting its weight to zero.
        """
        if self.recomb_mass_index is not None:
            self.recomb_mass_index[u.label].set_value(u.index, 0)
        if self.gc_mass_index is not None:
            self.gc_mass_index[u.label].set_value(u.index, 0)
        self.segment_stack.append(u)

    def free_hull(self, u):
        """
        Frees the specified hull making it ready for reuse.
        """
        u.left = None
        u.right = None
        u.ancestor_node = None
        u.insertion_order = math.inf
        self.hull_stack.append(u)

    def store_node(self, population, flags=0):
        self.flush_edges()
        return self.tables.nodes.add_row(
            time=self.t, flags=flags, population=population
        )

    def flush_edges(self):
        """
        Flushes the edges in the edge buffer to the table, squashing any adjacent edges.
        """
        if len(self.edge_buffer) > 0:
            self.edge_buffer.sort(key=lambda e: (e.child, e.left))
            left = self.edge_buffer[0].left
            right = self.edge_buffer[0].right
            child = self.edge_buffer[0].child
            parent = self.edge_buffer[0].parent
            for e in self.edge_buffer[1:]:
                assert e.parent == parent
                if e.left != right or e.child != child:
                    self.tables.edges.add_row(left, right, parent, child)
                    left = e.left
                    child = e.child
                right = e.right
            self.tables.edges.add_row(left, right, parent, child)
            self.edge_buffer = []

    def update_node_flag(self, node_id, flag):
        node_obj = self.tables.nodes[node_id]
        node_obj = node_obj.replace(flags=node_obj.flags | flag)
        self.tables.nodes[node_id] = node_obj

    def store_edge(self, left, right, parent, child):
        """
        Stores the specified edge to the output tree sequence.
        """
        if len(self.edge_buffer) > 0:
            last_edge = self.edge_buffer[-1]
            if last_edge.parent != parent:
                self.flush_edges()

        self.edge_buffer.append(
            tskit.Edge(left=left, right=right, parent=parent, child=child)
        )

    def finalise(self):
        """
        Finalises the simulation returns an msprime tree sequence object.
        """
        self.flush_edges()

        # Insert unary edges for any remainining lineages.
        current_time = self.t
        for population in self.P:
            for ancestor in population.iter_ancestors():
                node = tskit.NULL
                # See if there is already a node in this ancestor at the
                # current time
                seg = ancestor
                while seg is not None:
                    if self.tables.nodes[seg.node].time == current_time:
                        node = seg.node
                        break
                    seg = seg.next
                if node == tskit.NULL:
                    # Add a new node for the current ancestor
                    node = self.tables.nodes.add_row(
                        flags=0, time=current_time, population=population.id
                    )
                # Add in edges pointing to this ancestor
                seg = ancestor
                while seg is not None:
                    if seg.node != node:
                        self.tables.edges.add_row(seg.left, seg.right, node, seg.node)
                    seg = seg.next

        # Need to work around limitations in tskit Python API to prevent
        # individuals from getting unsorted:
        # https://github.com/tskit-dev/tskit/issues/1726
        ind_col = self.tables.nodes.individual
        ind_table = self.tables.individuals.copy()
        self.tables.sort()
        self.tables.individuals.clear()
        for ind in ind_table:
            self.tables.individuals.append(ind)
        self.tables.nodes.individual = ind_col
        return self.tables.tree_sequence()

    def simulate(self, end_time=None):
        if end_time is None:
            end_time = np.inf
        verify.verify(self)
        ret = self.simulate_smc(end_time)

        if ret == 2:  # _msprime.EXIT_MAX_TIME:
            self.t = end_time
        return self.finalise()

    def get_potential_destinations(self):
        """
        For each population return the set of populations for which it has a
        non-zero migration into.
        """
        N = len(self.P)
        potential_destinations = [set() for _ in range(N)]
        for j in range(N):
            for k in range(N):
                if self.migration_matrix[j][k] > 0:
                    potential_destinations[j].add(k)
        return potential_destinations

    def get_total_recombination_rate(self, label):
        total_rate = 0
        if self.recomb_mass_index is not None:
            total_rate = self.recomb_mass_index[label].get_total()
        return total_rate

    def get_total_gc_rate(self, label):
        total_rate = 0
        if self.gc_mass_index is not None:
            total_rate = self.gc_mass_index[label].get_total()
        return total_rate

    def get_total_gc_left_rate(self, label):
        gc_left_total = self.get_total_gc_left(label)
        return gc_left_total

    def get_total_gc_left(self, label):
        gc_left_total = 0
        num_ancestors = sum(pop.get_num_ancestors() for pop in self.P)
        mean_gc_rate = self.gc_map.mean_rate
        gc_left_total = num_ancestors * mean_gc_rate * self.tract_length
        return gc_left_total

    def find_cleft_individual(self, label, cleft_value):
        mean_gc_rate = self.gc_map.mean_rate
        individual_index = math.floor(cleft_value / (mean_gc_rate * self.tract_length))
        for pop in self.P:
            num_ancestors = pop.get_num_ancestors()
            if individual_index < num_ancestors:
                return pop._ancestors[label][individual_index]
            individual_index -= num_ancestors
        raise AssertionError()

    def simulate_smc(self, end_time):
        """
        Simulates the SMC(k) until all loci have coalesced.
        """
        ret = 0
        infinity = sys.float_info.max
        non_empty_pops = {pop.id for pop in self.P if pop.get_num_ancestors() > 0}
        potential_destinations = self.get_potential_destinations()

        # only worried about label 0 below
        while self.assert_stop_condition():
            verify.verify(self)
            # self.print_state()
            re_rate = self.get_total_recombination_rate(label=0)
            t_re = infinity
            if re_rate > 0:
                t_re = self.rng.expovariate(re_rate)

            # Gene conversion can occur within segments ..
            gc_rate = self.get_total_gc_rate(label=0)
            t_gcin = infinity
            if gc_rate > 0:
                t_gcin = self.rng.expovariate(gc_rate)
            # ... or to the left of the first segment.
            gc_left_rate = self.get_total_gc_left_rate(label=0)
            t_gc_left = infinity
            if gc_left_rate > 0:
                t_gc_left = self.rng.expovariate(gc_left_rate)

            # Common ancestor events occur within demes.
            t_ca = infinity
            for index in non_empty_pops:
                pop = self.P[index]
                assert pop.get_num_ancestors() > 0
                t = pop.get_common_ancestor_waiting_time(self.t, self.rng)
                if t < t_ca:
                    t_ca = t
                    ca_population = index
            t_mig = infinity
            # Migration events happen at the rates in the matrix.
            for j in non_empty_pops:
                source_size = self.P[j].get_num_ancestors()
                assert source_size > 0
                # for k in range(len(self.P)):
                for k in potential_destinations[j]:
                    rate = source_size * self.migration_matrix[j][k]
                    assert rate > 0
                    t = self.rng.expovariate(rate)
                    if t < t_mig:
                        t_mig = t
                        mig_source = j
                        mig_dest = k
            min_time = min(t_re, t_ca, t_gcin, t_gc_left, t_mig)
            assert min_time != infinity
            if self.t + min_time > end_time:
                ret = 2  # _msprime.MAX_EVENT_TIME
                break
            if self.t + min_time > self.modifier_events[0][0]:
                t, func, args = self.modifier_events.pop(0)
                self.t = t
                func(*args)
                # Don't bother trying to maintain the non-zero lists
                # through demographic events, just recompute them.
                non_empty_pops = {
                    pop.id for pop in self.P if pop.get_num_ancestors() > 0
                }
                potential_destinations = self.get_potential_destinations()
                event = "MOD"
            else:
                self.t += min_time
                if min_time == t_re:
                    event = "RE"
                    self.hudson_recombination_event(0)
                elif min_time == t_gcin:
                    event = "GCI"
                    self.wiuf_gene_conversion_within_event(0)
                elif min_time == t_gc_left:
                    event = "GCL"
                    self.wiuf_gene_conversion_left_event(0)
                elif min_time == t_ca:
                    event = "CA"
                    self.common_ancestor_event(ca_population, 0)
                    if self.P[ca_population].get_num_ancestors() == 0:
                        non_empty_pops.remove(ca_population)
                else:
                    event = "MIG"
                    self.migration_event(mig_source, mig_dest)
                    if self.P[mig_source].get_num_ancestors() == 0:
                        non_empty_pops.remove(mig_source)
                    assert self.P[mig_dest].get_num_ancestors() > 0
                    non_empty_pops.add(mig_dest)

            X = {pop.id for pop in self.P if pop.get_num_ancestors() > 0}
            assert non_empty_pops == X

        return ret

    def store_arg_edges(self, segment, u=-1):
        if u == -1:
            u = len(self.tables.nodes) - 1
        # Store edges pointing to current node to the left
        x = segment
        while x is not None:
            if x.node != u:
                self.store_edge(x.left, x.right, u, x.node)
            x.node = u
            x = x.prev
        # Store edges pointing to current node to the right
        x = segment
        while x is not None:
            if x.node != u:
                self.store_edge(x.left, x.right, u, x.node)
            x.node = u
            x = x.next

    def migration_event(self, j, k):
        """
        Migrates an individual from population j to population k.
        Only does label 0
        """
        label = 0
        source = self.P[j]
        dest = self.P[k]
        index = self.rng.randint(0, source.get_num_ancestors(label) - 1)
        x = source._ancestors[label][index]
        hull = x.get_hull()
        source.remove(x, label, hull)
        dest.add(x, label, hull)
        if self.additional_nodes.value & msprime.NODE_IS_MIG_EVENT > 0:
            self.store_node(k, flags=msprime.NODE_IS_MIG_EVENT)
            self.store_arg_edges(x)
        # Set the population id for each segment also.
        u = x
        while u is not None:
            u.population = k
            u = u.next

    def get_recomb_left_bound(self, seg):
        """
        Returns the left bound for genomic region over which the specified
        segment represents recombination events.
        """
        if seg.prev is None:
            left_bound = seg.left + 1 if self.discrete_genome else seg.left
        else:
            left_bound = seg.prev.right
        return left_bound

    def get_gc_left_bound(self, seg):
        # TODO remove me
        return self.get_recomb_left_bound(seg)

    def set_segment_mass(self, seg):
        """
        Sets the mass for the specified segment. All links *must* be
        appropriately set before calling this function.
        """
        if self.recomb_mass_index is not None:
            mass_index = self.recomb_mass_index[seg.label]
            recomb_left_bound = self.get_recomb_left_bound(seg)
            recomb_mass = self.recomb_map.mass_between(recomb_left_bound, seg.right)
            mass_index.set_value(seg.index, recomb_mass)
        if self.gc_mass_index is not None:
            mass_index = self.gc_mass_index[seg.label]
            gc_left_bound = self.get_gc_left_bound(seg)
            gc_mass = self.gc_map.mass_between(gc_left_bound, seg.right)
            mass_index.set_value(seg.index, gc_mass)

    def set_labels(self, segment, new_label):
        """
        Move the specified segment to the specified label.
        """
        mass_indexes = [self.recomb_mass_index, self.gc_mass_index]
        while segment is not None:
            masses = []
            for mass_index in mass_indexes:
                if mass_index is not None:
                    masses.append(mass_index[segment.label].get_value(segment.index))
                    mass_index[segment.label].set_value(segment.index, 0)
            segment.label = new_label
            for mass, mass_index in zip(masses, mass_indexes):
                if mass_index is not None:
                    mass_index[segment.label].set_value(segment.index, mass)
            segment = segment.next

    def choose_breakpoint(self, mass_index, rate_map):
        assert mass_index.get_total() > 0
        random_mass = self.rng.uniform(0, mass_index.get_total())
        y = self.segments[mass_index.find(random_mass)]
        y_cumulative_mass = mass_index.get_cumulative_sum(y.index)
        y_right_mass = rate_map.position_to_mass(y.right)
        bp_mass = y_right_mass - (y_cumulative_mass - random_mass)
        bp = rate_map.mass_to_position(bp_mass)
        if self.discrete_genome:
            bp = math.floor(bp)
        y_index = y.get_left_index()
        return y, bp

    def hudson_recombination_event(self, label, return_heads=False, y=None, bp=None):
        """
        Implements a recombination event.
        """
        self.num_re_events += 1
        if y is None or bp is None:
            y, bp = self.choose_breakpoint(
                self.recomb_mass_index[label], self.recomb_map
            )
        x = y.prev
        if y.left < bp:
            #   x         y
            # =====  ===|====  ...
            #          bp
            # becomes
            #   x     y
            # =====  ===  α        (LHS)
            #           =====  ... (RHS)
            alpha = self.copy_segment(y)
            alpha.left = bp
            alpha.prev = None
            if y.next is not None:
                y.next.prev = alpha
            y.next = None
            y.right = bp
            self.set_segment_mass(y)
            lhs_tail = y
        else:
            #   x            y
            # =====  |   =========  ...
            #
            # becomes
            #   x
            # =====          α          (LHS)
            #            =========  ... (RHS)
            x.next = None
            y.prev = None
            alpha = y
            lhs_tail = x

        # modify original hull
        pop = alpha.population
        lhs_hull = lhs_tail.get_hull()
        rhs_right = lhs_hull.right
        lhs_hull.right = min(lhs_tail.right + self.hull_offset, self.L)
        self.P[pop].reset_hull_right(label, lhs_hull, rhs_right, lhs_hull.right)

        # logic for alpha
        # create hull for alpha
        alpha_hull = self.alloc_hull(alpha.left, rhs_right - self.hull_offset, alpha)
        self.set_segment_mass(alpha)

        self.P[alpha.population].add(alpha, label, alpha_hull)
        if self.additional_nodes.value & msprime.NODE_IS_RE_EVENT > 0:
            self.store_node(lhs_tail.population, flags=msprime.NODE_IS_RE_EVENT)
            self.store_arg_edges(lhs_tail)
            self.store_node(alpha.population, flags=msprime.NODE_IS_RE_EVENT)
            self.store_arg_edges(alpha)
        ret = None
        if return_heads:
            x = lhs_tail
            # Seek back to the head of the x chain
            while x.prev is not None:
                x = x.prev
            ret = x, alpha
        return ret

    def generate_gc_tract_length(self):
        # generate tract length
        np_rng = np.random.default_rng(self.rng.randint(1, 2**16))
        if self.discrete_genome:
            tl = np_rng.geometric(1 / self.tract_length)
        else:
            tl = np_rng.exponential(self.tract_length)
        return tl

    def wiuf_gene_conversion_within_event(
        self, label, y=None, left_breakpoint=None, tl=None
    ):
        """
        Implements a gene conversion event that starts within a segment
        """
        # TODO This is more complicated than it needs to be now because
        # we're not trying to simulate the full GC process with this
        # one event anymore. Look into what bits can be dropped now
        # that we're simulating gc_left separately again.
        if y is None or left_breakpoint is None:
            y, left_breakpoint = self.choose_breakpoint(
                self.gc_mass_index[label], self.gc_map
            )
        x = y.prev
        # generate tract_length
        if tl is None:
            tl = self.generate_gc_tract_length()
            # print('tract length:', tl)
        assert tl > 0
        right_breakpoint = min(left_breakpoint + tl, self.L)
        if y.left >= right_breakpoint:
            #                  y
            # ...  |   |   ========== ...
            #     lbp rbp
            return None
        self.num_gc_events += 1
        pop = y.population
        hull = y.get_hull()
        reset_right = -1
        # Process left break
        insert_alpha = True
        if left_breakpoint <= y.left:
            #  x             y
            # =====  |  ==========
            #       lbp
            #
            # becomes
            #  x
            # =====         α
            #           ==========
            if x is None:
                raise NotImplementedError
                # In this case we *don't* insert alpha because it is already
                # the head of a segment chain
                insert_alpha = False
            else:
                x.next = None
                reset_right = x.right
            y.prev = None
            alpha = y
            tail = x
        else:
            #  x             y
            # =====     ====|=====
            #              lbp
            #
            # becomes
            #  x         y
            # =====     ====   α
            #               ======
            alpha = self.copy_segment(y)
            alpha.left = left_breakpoint
            alpha.prev = None
            if y.next is not None:
                y.next.prev = alpha
            y.next = None
            y.right = left_breakpoint
            self.set_segment_mass(y)
            tail = y
            reset_right = left_breakpoint
        self.set_segment_mass(alpha)

        # Find the segment z that the right breakpoint falls in
        z = alpha
        hull_left = z.left
        while z is not None and right_breakpoint >= z.right:
            hull_right = z.right
            z = z.next

        head = None
        # Process the right break
        if z is not None:
            if z.left < right_breakpoint:
                #   tail             z
                # ======
                #       ...  ===|==========
                #              rbp
                #
                # becomes
                #  tail              head
                # =====         ===========
                #      ...   ===
                #             z
                head = self.copy_segment(z)
                head.left = right_breakpoint
                if z.next is not None:
                    z.next.prev = head
                z.right = right_breakpoint
                z.next = None
                self.set_segment_mass(z)
                hull_right = right_breakpoint
            else:
                #   tail             z
                # ======
                #   ...   |   =============
                #        rbp
                #
                # becomes
                #  tail             z
                # ======      =============
                #  ...
                if z.prev is not None:
                    z.prev.next = None
                head = z
            if tail is not None:
                tail.next = head
            head.prev = tail
            self.set_segment_mass(head)
        else:
            # rbp lies beyond end of segment chain
            # regular recombination logic applies
            if insert_alpha:
                self.P[pop].reset_hull_right(
                    label, hull, hull_right + self.hull_offset, reset_right
                )

        #        y            z
        #  |  ========== ... ===== |
        # lbp                     rbp
        # When y and z are the head and tail of the segment chains, then
        # this GC event does nothing. This logic takes care of this situation.
        new_individual_head = None
        if insert_alpha:
            new_individual_head = alpha
        elif head is not None:
            new_individual_head = head
        if new_individual_head is not None:
            assert hull_left < hull_right
            hull = self.alloc_hull(hull_left, hull_right, new_individual_head)
            self.P[new_individual_head.population].add(
                new_individual_head, new_individual_head.label, hull
            )

    def wiuf_gene_conversion_left_event(self, label):
        """
        Implements a gene conversion event that started left of a first segment.
        """
        random_gc_left = self.rng.uniform(0, self.get_total_gc_left(label))
        # Get segment where gene conversion starts from left
        y = self.find_cleft_individual(label, random_gc_left)
        assert y is not None

        # generate tract_length
        tl = self.generate_gc_tract_length()

        bp = y.left + tl
        while y is not None and y.right <= bp:
            y = y.next

        # if the gene conversion spans the whole individual nothing happens
        if y is None:
            #    last segment
            # ... ==========   |
            #                  bp
            # stays in current state
            return None

        self.num_gc_events += 1
        x = y.prev
        pop = y.population
        lhs_hull = y.get_hull()
        if y.left < bp:
            #  x          y
            # =====   =====|====
            #              bp
            # becomes
            #  x         y
            # =====   =====
            #              =====
            #                α
            alpha = self.copy_segment(y)
            alpha.left = bp
            alpha.prev = None
            if alpha.next is not None:
                alpha.next.prev = alpha
            y.next = None
            y.right = bp
            self.set_segment_mass(y)
            right = y.right
        else:
            #  x          y
            # ===== |  =========
            #       bp
            # becomes
            #  x
            # =====
            #          =========
            #              α
            # split the link between x and y.
            x.next = None
            y.prev = None
            alpha = y
            right = x.right
        # lhs
        # logic is identical to the lhs recombination event
        rhs_right = lhs_hull.right
        lhs_hull.right = right + self.hull_offset
        self.P[pop].reset_hull_right(label, lhs_hull, rhs_right, lhs_hull.right)

        # rhs
        self.set_segment_mass(alpha)
        hull = self.alloc_hull(alpha.left, rhs_right, alpha)
        self.P[alpha.population].add(alpha, label, hull)

    def store_additional_nodes_edges(self, flag, new_node_id, z):
        if self.additional_nodes.value & flag > 0:
            if new_node_id == -1:
                new_node_id = self.store_node(z.population, flags=flag)
            else:
                self.update_node_flag(new_node_id, flag)
            self.store_arg_edges(z, new_node_id)
        return new_node_id

    def defrag_segment_chain(self, z):
        y = z
        while y.prev is not None:
            x = y.prev
            if x.right == y.left and x.node == y.node:
                x.right = y.right
                x.next = y.next
                if y.next is not None:
                    y.next.prev = x
                self.set_segment_mass(x)
                self.free_segment(y)
            y = x

    def defrag_breakpoints(self):
        # Defrag the breakpoints set
        j = 0
        k = 0
        while k < self.L:
            k = self.S.succ_key(j)
            if self.S[j] == self.S[k]:
                del self.S[k]
            else:
                j = k

    def get_random_pair(self, pop, label, random_mass=None):
        # pick random pair
        if random_mass is None:
            num_pairs = self.P[pop].get_num_pairs(label)
            random_mass = self.rng.randint(1, num_pairs)
        mass_index = self.P[pop].coal_mass_index[label]

        # get first element of pair
        hull1_index = mass_index.find(random_mass)
        hull1_cumulative_mass = mass_index.get_cumulative_sum(hull1_index)
        remaining_mass = hull1_cumulative_mass - random_mass

        # get second element of pair
        avl = self.P[pop].hulls_left[label].avl
        hull1 = self.hulls[hull1_index]
        left = hull1.left
        hull2 = hull1
        while remaining_mass >= 0:
            hull2 = avl.prev_key(hull2)
            if hull2.left == left or hull2.right > left:
                remaining_mass -= 1

        return (hull1_index, hull2.index)

    def common_ancestor_event(self, population_index, label, random_pair=None):
        """
        Implements a coancestry event.
        """
        pop = self.P[population_index]
        # Choose two ancestors uniformly according to hulls_left weights
        if random_pair is None:
            random_pair = self.get_random_pair(population_index, label)
        hull_i_ptr, hull_j_ptr = random_pair
        hull_i = self.hulls[hull_i_ptr]
        hull_j = self.hulls[hull_j_ptr]
        x = hull_i.ancestor_node
        y = hull_j.ancestor_node
        pop.remove(x, label, hull_i)
        pop.remove(y, label, hull_j)
        self.free_hull(hull_i)
        self.free_hull(hull_j)
        self.merge_two_ancestors(population_index, label, x, y)

    def merge_two_ancestors(self, population_index, label, x, y, u=-1):
        pop = self.P[population_index]
        self.num_ca_events += 1
        z = None
        coalescence = False
        defrag_required = False
        while x is not None or y is not None:
            alpha = None
            if x is None or y is None:
                if x is not None:
                    alpha = x
                    x = None
                if y is not None:
                    alpha = y
                    y = None
            else:
                if y.left < x.left:
                    beta = x
                    x = y
                    y = beta
                if x.right <= y.left:
                    alpha = x
                    x = x.next
                    alpha.next = None
                elif x.left != y.left:
                    alpha = self.copy_segment(x)
                    alpha.prev = None
                    alpha.next = None
                    alpha.right = y.left
                    x.left = y.left
                else:
                    if not coalescence:
                        coalescence = True
                        if u == -1:
                            self.store_node(population_index)
                            u = len(self.tables.nodes) - 1
                    # Put in breakpoints for the outer edges of the coalesced
                    # segment
                    left = x.left
                    r_max = min(x.right, y.right)
                    if left not in self.S:
                        j = self.S.floor_key(left)
                        self.S[left] = self.S[j]
                    if r_max not in self.S:
                        j = self.S.floor_key(r_max)
                        self.S[r_max] = self.S[j]
                    # Update the number of extant segments.
                    min_overlap = 2
                    if self.stop_condition is not None:
                        min_overlap = 0
                    if self.S[left] == min_overlap:
                        self.S[left] = 0
                        right = self.S.succ_key(left)
                    else:
                        right = left
                        while right < r_max and self.S[right] != min_overlap:
                            self.S[right] -= 1
                            right = self.S.succ_key(right)
                        alpha = self.alloc_segment(
                            left=left,
                            right=right,
                            node=u,
                            population=population_index,
                            label=label,
                        )
                    if x.node != u:  # required for dtwf and fixed_pedigree
                        self.store_edge(left, right, u, x.node)
                    if y.node != u:  # required for dtwf and fixed_pedigree
                        self.store_edge(left, right, u, y.node)
                    # Now trim the ends of x and y to the right sizes.
                    if x.right == right:
                        self.free_segment(x)
                        x = x.next
                    else:
                        x.left = right
                    if y.right == right:
                        self.free_segment(y)
                        y = y.next
                    else:
                        y.left = right

            # loop tail; update alpha and integrate it into the state.
            if alpha is not None:
                if z is None:
                    # we do not know yet where the hull will end.
                    hull = self.alloc_hull(alpha.left, alpha.right, alpha)
                    pop.add(alpha, label, None)
                else:
                    if (coalescence and not self.coalescing_segments_only) or (
                        self.additional_nodes.value & msprime.NODE_IS_CA_EVENT > 0
                    ):
                        defrag_required |= z.right == alpha.left
                    else:
                        defrag_required |= (
                            z.right == alpha.left and z.node == alpha.node
                        )
                    z.next = alpha
                alpha.prev = z
                self.set_segment_mass(alpha)
                z = alpha

        if coalescence:
            if not self.coalescing_segments_only:
                self.store_arg_edges(z, u)
        else:
            if self.additional_nodes.value & msprime.NODE_IS_CA_EVENT > 0:
                self.store_additional_nodes_edges(msprime.NODE_IS_CA_EVENT, u, z)

        if defrag_required:
            self.defrag_segment_chain(z)
        if coalescence:
            self.defrag_breakpoints()

        # update right endpoint hull
        # surely this can be improved upon
        if z is not None:
            y = z
            while y is not None:
                y.hull = hull
                y = y.prev
            while z is not None:
                right = z.right
                z.hull = hull
                z = z.next
            hull.right = min(right + self.hull_offset, self.L)
            pop.add_hull(label, hull)

    def print_state(self):
        for pop in self.P:
            pop.print_state()
