import copy
import math
import numpy as np
import pytest
import tskit


import hulls.hulltracker as tracker
import hulls.verify as verify
import hulls.algorithm as alg

class TestAVL:
    @pytest.fixture(scope="class")
    def pre_defined_tables(self):
        # 2.00┊ 6 9 7 8 ┊  9  7 8 ┊  9  7 5 ┊ 911 7 5 ┊ 1011 7 5 ┊ 1011  5  ┊
        #     ┊ ┃ ┃ ┃ ┃ ┊  ┃  ┃ ┃ ┊  ┃  ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊ ┃  ┃ ┃ ┃ ┊ ┃  ┃ ┏┻┓ ┊
        # 1.00┊ ┃ 4 ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊ 4 ┃ ┃ ┃ ┊ ┃  ┃ ┃ ┃ ┊ ┃  ┃ ┃ ┃ ┊
        #     ┊ ┃ ┃ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊ ┃  ┃ ┃ ┃ ┊ ┃  ┃ ┃ ┃ ┊
        # 0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0  1 2 3 ┊ 0  1 2 3 ┊
        #     0        10        20        40        50         60         100

        tables = tskit.TableCollection(100)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.populations.add_row()
        ### nodes table
        tables.nodes.add_row(flags=1, time=0, population=0)  # 0
        tables.nodes.add_row(flags=1, time=0, population=0)  # 1
        tables.nodes.add_row(flags=1, time=0, population=0)  # 2
        tables.nodes.add_row(flags=1, time=0, population=0)  # 3
        tables.nodes.add_row(flags=0, time=1, population=0)  # 4
        tables.nodes.add_row(flags=0, time=2, population=0)  # 5
        tables.nodes.add_row(flags=0, time=2, population=0)  # 6
        tables.nodes.add_row(flags=0, time=2, population=0)  # 7
        tables.nodes.add_row(flags=0, time=2, population=0)  # 8
        tables.nodes.add_row(flags=0, time=2, population=0)  # 9
        tables.nodes.add_row(flags=0, time=2, population=0)  # 10
        tables.nodes.add_row(flags=0, time=2, population=0)  # 11

        # edges table
        tables.edges.add_row(left=10, right=50, parent=4, child=0)
        tables.edges.add_row(left=0, right=40, parent=4, child=1)
        tables.edges.add_row(left=20, right=100, parent=5, child=3)
        tables.edges.add_row(left=60, right=100, parent=5, child=2)
        tables.edges.add_row(left=0, right=10, parent=6, child=0)
        tables.edges.add_row(left=0, right=60, parent=7, child=2)
        tables.edges.add_row(left=0, right=20, parent=8, child=3)
        tables.edges.add_row(left=0, right=50, parent=9, child=4)
        tables.edges.add_row(left=50, right=100, parent=10, child=0)
        tables.edges.add_row(left=40, right=100, parent=11, child=1)
        tables.sort()
        return tables

    def test_avl(self):
        tables = tskit.TableCollection(100)
        tables.populations.add_row()
        sample_count = 10
        for _ in range(sample_count):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=0)
        sim = tracker.Simulator(
            initial_state=tables,
        )
        pairs = sim.P[0].get_num_pairs()
        assert pairs == math.comb(10, 2)


    def test_setup_simple(self, pre_defined_tables):
        tables = pre_defined_tables
        sim = tracker.Simulator(initial_state=tables)
        num_pairs = sim.P[0].get_num_pairs()
        assert num_pairs == 14
        random_pair = np.zeros(2, dtype=np.int64)
        pop = 0
        label = 0
        obs_pairs = set()
        for i in range(num_pairs):
            sim.P[label].get_random_pair(random_pair, i, label)
            obs_pairs.add(tuple(random_pair))
        # node_to_hull_idx = {5:100, 6:99, 7:98, 8:97, 9:96, 10:95, 11:94}
        all_random_pairs = set(
            [
                (97, 96),
                (98, 97),
                (98, 96),
                (99, 98),
                (99, 97),
                (99, 96),
                (100, 96),
                (100, 98),
                (94, 100),
                (94, 98),
                (94, 96),
                (95, 94),
                (95, 100),
                (95, 98),
            ]
        )
        assert obs_pairs == all_random_pairs
        # add in new hull
        old_state = copy.deepcopy(sim.P[0].hulls_left[0])
        left = 45
        right = 65
        seg_index = sim.alloc_segment(left, right, -1, pop)
        new_hull = sim.alloc_hull(left, right, seg_index)
        sim.P[0].add_hull(0, new_hull)
        assert sim.P[0].hulls_left[0][new_hull] == 4
        # remove this hull again
        sim.P[0].remove_hull(0, new_hull)
        sim.free_hull(new_hull)
        # should be restored to old state
        for key, value in sim.P[0].hulls_left[0].items():
            assert old_state[key] == value
        # add in new hull
        left = 20
        right = 60
        new_hull = sim.alloc_hull(left, right, seg_index)
        sim.P[0].add_hull(0, new_hull)
        assert sim.P[0].hulls_left[0][new_hull] == 2
        # remove this hull again
        sim.P[0].remove_hull(0, new_hull)
        sim.free_hull(new_hull)
        for key, value in sim.P[0].hulls_left[0].items():
            assert old_state[key] == value

    def test_coalescence_event_fixed(self, pre_defined_tables):
        tables = pre_defined_tables
        sim = tracker.Simulator(initial_state=tables)
        verify.verify_hulls(sim)
        #print('start state')
        #print(sim.P[0]._ancestors[0])
        #print(sim.P[0].hulls_left[0])
        self.t = 3.0
        pop = 0
        label= 0
        random_pair = np.array([98, 96], dtype=np.int64)
        sim.common_ancestor_event(pop, label, random_pair)
        #print('after coalescence event')
        #print(sim.P[0]._ancestors[0])
        #print(sim.P[0].hulls_left[0])
        verify.verify_hulls(sim)

    def test_random_coalescence_event(self, pre_defined_tables):
        tables = pre_defined_tables
        sim = tracker.Simulator(initial_state=tables)
        verify.verify_hulls(sim)
        #print('starting state')
        #print(sim.P[0]._ancestors[0])
        #print(sim.P[0].hulls_left[0])
        self.t = 3.0
        pop = 0
        label= 0
        sim.common_ancestor_event(pop, label)
        #print('after coalescence event')
        #print(sim.P[0]._ancestors[0])
        #print(sim.P[0].hulls_left[0])
        verify.verify_hulls(sim)
        verify.verify(sim)

    def test_recombination(self, pre_defined_tables):
        tables = pre_defined_tables
        sim = tracker.Simulator(initial_state=tables, recombination_rate=0.1)
        label = 0
        #print(sim.P[0]._ancestors[0])
        #print(sim.P[0].hulls_left[0])
        for _ in range(10):
            sim.hudson_recombination_event(label)
            #print(sim.P[0]._ancestors[0])
            #print(sim.P[0].hulls_left[0])
            verify.verify_hulls(sim)



def make_initial_state(sample_configuration, sequence_length):
    tables = tskit.TableCollection(sequence_length)
    for pop_id, sample_count in enumerate(sample_configuration):
        tables.populations.add_row()
        for _ in range(sample_count):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, population=pop_id)
