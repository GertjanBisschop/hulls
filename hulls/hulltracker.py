import bintrees
import math
import msprime
import numpy as np
import random
import tskit

import hulls.algorithm as alg
import hulls.verify as verify


class Hull:
    def __init__(self, index):
        self.left = None
        self.right = None
        self.ancestor_node = None
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
        hull_offset=0,
        coalescing_segments_only=True,
        additional_nodes=None,
        random_seed=None,
    ):
        N = 1  # num pops
        self.num_labels = 1
        self.num_populations = N
        population_sizes = [10_000 for _ in range(self.num_populations)]
        population_growth_rates = [0 for _ in range(self.num_populations)]
        if initial_state is None:
            raise ValueError("Requires an initial state.")
        if isinstance(initial_state, tskit.TreeSequence):
            initial_state = initial_state.dump_tables()
        self.tables = initial_state
        self.L = int(initial_state.sequence_length)
        self.rng = random.Random(random_seed)

        self.recomb_map = alg.RateMap([0, self.L], [recombination_rate, 0])
        self.gc_map = alg.RateMap([0, self.L], [gene_conversion_rate, 0])
        self.tract_length = gene_conversion_length
        self.discrete_genome = True

        self.max_segments = max_segments
        self.max_hulls = max_segments
        self.hull_offset = hull_offset
        self.segment_stack = []
        self.segments = [None for _ in range(self.max_segments + 1)]
        for j in range(self.max_segments):
            s = alg.Segment(j + 1)
            self.segments[j + 1] = s
            self.segment_stack.append(s)
        self.hull_stack = []
        # double check whether this is maintained/used.
        self.hulls = [None for _ in range(self.max_hulls + 1)]
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
        self.pairwise_count_index = [
            alg.FenwickTree(self.max_hulls) for j in range(self.num_labels)
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
                self.P[pop].hulls_left[label][new_hull] = -1

        # initialise the correct coalesceable pairs count
        for pop in self.P:
            for label, avl_tree in enumerate(pop.hulls_left):
                ranks_left = pop.hulls_left_rank[label]
                ranks_right = pop.hulls_right_rank[label]
                count = 0
                for hull in avl_tree.keys():
                    num_ending_before_hull = ranks_right.get_cumulative_sum(
                        hull.left + 1
                    )
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

    def alloc_hull(self, left, right, ancestor_node):
        hull = self.hull_stack.pop()
        hull.left = int(left)
        hull.right = min(int(right) + self.hull_offset, self.L)
        hull.ancestor_node = ancestor_node
        assert ancestor_node.prev is None
        ancestor_node.hull = hull
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

    def free_hull(self, u):
        """
        Frees the specified hull making it ready for reuse.
        """
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

    def simulate(self, end_time):
        verify.verify(sim)
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
            verify.verify(sim)
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
                    self.common_ancestor_event(lhs_hullca_population, 0)
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
        x = source.remove(index, label)
        dest.add(x, label)
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

    def reset_hull_right(self, hull, old_right, new_right, pop, label):
        # when resetting the hull.right of a pre-existing hull we need to
        # decrement count of all lineages starting off between hull.left and bp
        avl = self.P[pop].hulls_left[label]
        max_hull = avl.max_key()
        curr_hull = hull
        while curr_hull < max_hull:
            curr_hull = avl.succ_key(curr_hull)
            if curr_hull.left >= old_right:
                break
            if curr_hull.left >= new_right:
                avl[curr_hull] -= 1

        # adjust rank of hull.right
        self.P[pop].hulls_right_rank[label].increment(int(old_right) + 1, -1)
        self.P[pop].hulls_right_rank[label].increment(int(new_right) + 1, 1)

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
        # print("segment:", y, "breakpoint:", bp)
        y_index = y.get_left_index()
        # print("segment chain left", y.get_left_end(), y.get_left_index())
        # print(alg.Segment.show_chain(self.segments[y_index]))
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
        lhs_hull.right = lhs_tail.right + self.hull_offset
        self.reset_hull_right(lhs_hull, rhs_right, lhs_hull.right, pop, label)
        # logic for alpha
        self.set_segment_mass(alpha)
        # create hull for alpha
        alpha_hull = self.alloc_hull(alpha.left, rhs_right - self.hull_offset, alpha)
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
            tl = int(self.generate_gc_tract_length())
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
                self.reset_hull_right(hull, hull_right, reset_right, pop, label)

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
        lhs_hull.right = int(right + self.hull_offset)
        self.reset_hull_right(lhs_hull, rhs_right, lhs_hull.right, pop, label)

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

    def common_ancestor_event(self, population_index, label, random_pair=None):
        """
        Implements a coancestry event.
        """
        pop = self.P[population_index]
        # Choose two ancestors uniformly according to hulls_left weights
        if random_pair is None:
            random_pair = -np.ones(2, dtype=np.int64)
            num_pairs = pop.get_num_pairs()
            random_count = self.rng.randint(0, num_pairs - 1)
            pop.get_random_pair(random_pair, random_count, label)
        hull_i_ptr, hull_j_ptr = random_pair
        hull_i = self.hulls[hull_i_ptr]
        hull_j = self.hulls[hull_j_ptr]
        x = hull_i.ancestor_node
        y = hull_j.ancestor_node
        # print(f"random pair: {random_pair}")
        # print(f"coalescing:{x} + {y}")
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
        right = z.get_right_end()
        hull.right = min(int(right) + self.hull_offset, self.L)
        pop.add_hull(label, hull)

    def print_state(self):
        for pop in self.P:
            pop.print_state()
