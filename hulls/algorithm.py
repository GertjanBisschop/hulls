import bintrees
import math
import numpy as np
import random
import sys


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

    def __repr__(self):
        return repr((self.left, self.right, self.node))

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
        while seg is not None:
            left = seg.left
            seg = seg.prev

        return left

    def get_right_end(self):
        while seg is not None:
            right = seg.right
            seg = seg.next

        return right


class Population:
    """
    Class representing a population in the simulation.
    """

    def __init__(self, id_, num_labels=1):
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
        # track all class::Hull
        self.hulls_left = [bintrees.AVLTree() for _ in range(num_labels)]
        # track rank of hulls_left
        self.hulls_left_rank = [None for _ in range(num_labels)]
        # track rank of hulls_right
        self.hulls_right_rank = [None for _ in range(num_labels)]
        self.num_pairs = np.zeros(num_labels, dtype=np.uint64)

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
            return sum(
                sum(count for count in avl_tree.values())
                for avl_tree in self.hulls_left
            )
        else:
            return sum(count for count in self.hulls_left[label].values())

    def get_size(self, t):
        """
        Returns the size of this population at time t.
        """
        dt = t - self.start_time
        return self.start_size * math.exp(-self.growth_rate * dt)

    def get_common_ancestor_waiting_time(self, t):
        """
        Returns the random waiting time until a common ancestor event
        occurs within this population.
        """
        ret = sys.float_info.max
        u = self.get_num_pairs()
        if u > 1:
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

    def get_random_pair(self, random_pair, random_count, label):
        avl = self.hulls_left[label]
        # pick first lineage
        # by traversing the avl tree until
        # the cumulative count of the keys (coalesceable pairs)
        # matches random_count
        for hull, pairs_count in avl.items():
            if random_count < pairs_count:
                random_pair[0] = hull.index
                break
            else:
                random_count -= pairs_count
        left = hull.left

        # pick second lineage
        # traverse avl_tree towards smallest element until we
        # find the random_count^th element that can coalesce with
        # the first picked hull.
        while random_count >= 0:
            hull = avl.prev_key(hull)
            if hull.left == left or hull.right > left:
                random_count -= 1
        random_pair[-1] = hull.index

    def get_ind_range(self, t):
        """Returns ind labels at time t"""
        first_ind = np.sum([self.get_size(t_prev) for t_prev in range(0, int(t))])
        last_ind = first_ind + self.get_size(t)

        return range(int(first_ind), int(last_ind) + 1)

    def remove_hull(self, label, hull):
        count = self.hulls_left[label].pop(hull)
        # self.num_pairs[label] -= count
        # decrement rank
        self.hulls_left_rank[label].increment(hull.left + 1, -1)
        self.hulls_right_rank[label].increment(hull.right + 1, -1)

    def remove(self, individual, label=0, hull=None):
        """
        Removes and returns the individual at the specified index.
        """
        # update hull information
        if hull is not None:
            self.remove_hull(label, hull)
        return self._ancestors[label].remove(individual)

    def remove_individual(self, individual, label=0):
        """
        Removes the given individual from its population.
        """
        return self._ancestors[label].remove(individual)

    def add_hull(self, label, hull):
        left = hull.left
        num_ending_before_left = self.hulls_right_rank[label].get_cumulative_sum(
            hull.left + 1
        )
        num_starting_after_left = self.hulls_left_rank[label].get_cumulative_sum(
            hull.left + 1
        )
        count = num_starting_after_left - num_ending_before_left
        self.hulls_left[label][hull] = count
        # correction is needed because the rank implementation in the Python version
        # assumes that new hulls are added below hulls with the same starting point.
        correction = 0
        curr_hull = self.hulls_left[label].prev_key(hull)
        while curr_hull.left == left:
            correction += 1
        self.hulls_left[label][hull] -= correction
        # self.num_pairs[label] += count - correction
        # increment rank
        self.hulls_left_rank[label].increment(hull.left + 1, 1)
        self.hulls_right_rank[label].increment(hull.right + 1, 1)

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
