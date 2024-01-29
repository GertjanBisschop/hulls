import bintrees
import numpy as np

import hulls.algorithm as alg


class Hull:
    def __init__(self, index):
        self.left = None
        self.right = None
        self.segment_ptr = None
        # index should reflect insertion order
        self.index = index
        self.count = 0

    def __lt__(self, other):
        return (self.left, self.index) < (other.left, other.index)

    def __repr__(self):
        return f"{self.left}, {self.right}, {self.index}"


class Simulator:
    def __init__(
        self, 
        *,
        initial_state=None, 
        migration_map=None,
        max_segments=100,
        recombination_rate=0.0,
        gene_conversion_rate=0.0,
        gene_conversion_length=1,
        stop_condition=None, 
        hull_offset=0
    ):
        N = 1  # num pops
        self.num_labels = 1
        self.num_populations = N
        population_sizes = [10_000 for _ in range(self.num_populations)]
        population_growth_rates = [0 for _ in range(self.num_populations)]
        if initial_state is None:
            raise ValueError('Requires an initial state.')
        self.L = int(initial_state.sequence_length)
        
        self.recomb_map = alg.RateMap([0, self.L], [recombination_rate, 0])
        self.gc_map = alg.RateMap([0, self.L], [gene_conversion_rate, 0])
        self.tract_length = gene_conversion_length
        self.discrete_genome = True

        self.max_segments = max_segments
        self.hull_offset = hull_offset
        self.segment_stack = []
        self.segments = [None for j in range(self.max_segments + 1)]
        for j in range(self.max_segments):
            s = alg.Segment(j + 1)
            self.segments[j + 1] = s
            self.segment_stack.append(s)
        self.hull_stack = []
        self.hulls = [None for j in range(self.max_segments + 1)]
        for j in range(self.max_segments):
            h = Hull(j + 1)
            self.hulls[j + 1] = h
            self.hull_stack.append(h)
        self.S = bintrees.AVLTree()
        self.P = [alg.Population(id_, self.num_labels) for id_ in range(N)]
        for pop in self.P:
            for i in range(self.num_labels):
                pop.hulls_left_rank[i] = alg.FenwickTree(self.L + 1)
                pop.hulls_right_rank[i] = alg.FenwickTree(self.L + 1)
        if self.recomb_map.total_mass == 0:
            self.recomb_mass_index = None
        else:
            self.recomb_mass_index = [
                alg.FenwickTree(self.max_segments) for j in range(self.num_labels)
            ]
        if self.gc_map.total_mass == 0:
            self.gc_mass_index = None
        else:
            self.gc_mass_index = [
                alg.FenwickTree(self.max_segments) for j in range(self.num_labels)
            ]
        self.S = bintrees.AVLTree()
        for pop in self.P:
            pop.set_start_size(population_sizes[pop.id])
            pop.set_growth_rate(population_growth_rates[pop.id], 0)
        self.edge_buffer = []

        self.initialise(initial_state)

        self.num_ca_events = 0
        self.num_re_events = 0
        self.num_gc_events = 0

        self.stop_condition = stop_condition

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
                seg_ptr = seg.index
                pop = seg.population
                label = seg.label
                self.P[pop].add(seg)
                while seg is not None:
                    # self.set_segment_mass(seg)
                    right_end = seg.right
                    seg = seg.next
                new_hull = self._alloc_hull(left_end, right_end, seg_ptr)
                self.P[pop].hulls_left[label][new_hull] = -1

        # initialise the correct coalesceable pairs count
        for pop in self.P:
            for label, avl_tree in enumerate(pop.hulls_left): 
                ranks_left = pop.hulls_left_rank[label]
                ranks_right = pop.hulls_right_rank[label]        
                count = 0
                for hull in avl_tree.keys():
                    num_ending_before_hull = ranks_right.get_cumulative_sum(hull.left + 1)
                    ranks_left.increment(hull.left + 1, 1)
                    ranks_right.increment(hull.right + 1, 1)
                    avl_tree[hull] = count - num_ending_before_hull
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

    def insert_hull(self, pop, label, hull):
        # insert left
        avl_tree = self.P[pop].hulls_left[label]
        ranks_left = self.P[pop].hulls_left_rank[label]
        ranks_right = self.P[pop].hulls_right_rank[label]
        if hull.left == 0:
            count = ranks_left.get_cumulative_sum(hull.left + 1)
        else:
            num_ending_before_hull = ranks_right.get_cumulative_sum(hull.left)
            num_starting_after_hull = ranks_left.get_cumulative_sum(hull.left + 1)
            count = num_starting_after_hull - num_ending_before_hull
        avl_tree[hull] = count
        # update ranks
        self.P[pop].hulls_left_rank[label].increment(hull.left + 1, 1)
        self.P[pop].hulls_right_rank[label].increment(hull.right + 1, 1)
        
    def remove_hull(self):
        pass

    def _alloc_hull(self, left, right, segment_ptr):
        hull = self.hull_stack.pop()
        hull.left = int(left)
        hull.right = int(right) + self.hull_offset
        hull.segment_ptr = segment_ptr

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

    def print_state(self):
        for pop in self.P:
            pop.print_state()

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

    def run_simulate(self, end_time):
        # self.verify()
        ret = self.simulate(end_time)
        
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

    def simulate(self, end_time):
        """
        Simulates the SMC(k) until all loci have coalesced.
        """
        ret = 0
        infinity = sys.float_info.max
        non_empty_pops = {pop.id for pop in self.P if pop.get_num_ancestors() > 0}
        potential_destinations = self.get_potential_destinations()

        # only worried about label 0 below
        while self.assert_stop_condition():
            self.verify()
            # self.print_state()
            re_rate = self.get_total_recombination_rate(label=0)
            t_re = infinity
            if re_rate > 0:
                t_re = random.expovariate(re_rate)

            # Gene conversion can occur within segments ..
            gc_rate = self.get_total_gc_rate(label=0)
            t_gcin = infinity
            if gc_rate > 0:
                t_gcin = random.expovariate(gc_rate)
            # ... or to the left of the first segment.
            gc_left_rate = self.get_total_gc_left_rate(label=0)
            t_gc_left = infinity
            if gc_left_rate > 0:
                t_gc_left = random.expovariate(gc_left_rate)

            # Common ancestor events occur within demes.
            t_ca = infinity
            for index in non_empty_pops:
                pop = self.P[index]
                assert pop.get_num_ancestors() > 0
                t = pop.get_common_ancestor_waiting_time(self.t)
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
                    t = random.expovariate(rate)
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
            logger.info(
                "%s time=%f n=%d",
                event,
                self.t,
                sum(pop.get_num_ancestors() for pop in self.P),
            )

            X = {pop.id for pop in self.P if pop.get_num_ancestors() > 0}
            assert non_empty_pops == X

        return ret