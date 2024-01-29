import math
import pytest
import tskit


import hulls.hulltracker as tracker

class TestAVL:

    def test_avl(self):
        tables = tskit.TableCollection(100)
        tables.populations.add_row()
        sample_count = 10
        for _ in range(sample_count):
            tables.nodes.add_row(
                flags=tskit.NODE_IS_SAMPLE, time=0, population=0
            )
        sim = tracker.Simulator(
            initial_state=tables,
        )
        sim.print_state()
        pairs = sim.P[0].get_num_pairs()
        assert pairs == math.comb(10,2) 

    def test_setup_simple(self):
        # 2.00┊ 6 9 7 8 ┊  9  7 8 ┊  9  7 5 ┊ 911 7 5 ┊ 1011 7 5 ┊ 1011  5  ┊ 
        #     ┊ ┃ ┃ ┃ ┃ ┊  ┃  ┃ ┃ ┊  ┃  ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊ ┃  ┃ ┃ ┃ ┊ ┃  ┃ ┏┻┓ ┊ 
        # 1.00┊ ┃ 4 ┃ ┃ ┊  4  ┃ ┃ ┊  4  ┃ ┃ ┊ 4 ┃ ┃ ┃ ┊ ┃  ┃ ┃ ┃ ┊ ┃  ┃ ┃ ┃ ┊ 
        #     ┊ ┃ ┃ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┏┻┓ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊ ┃  ┃ ┃ ┃ ┊ ┃  ┃ ┃ ┃ ┊ 
        # 0.00┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0  1 2 3 ┊ 0  1 2 3 ┊ 
        #     0        10        20        40        50         60         100

        tables = tskit.TableCollection(100)
        tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        
        ### nodes table
        tables.nodes.add_row(flags=1, time=0) # 0
        tables.nodes.add_row(flags=1, time=0) # 1
        tables.nodes.add_row(flags=1, time=0) # 2
        tables.nodes.add_row(flags=1, time=0) # 3
        tables.nodes.add_row(flags=0, time=1) # 4
        tables.nodes.add_row(flags=0, time=2) # 5
        tables.nodes.add_row(flags=0, time=2) # 6
        tables.nodes.add_row(flags=0, time=2) # 7
        tables.nodes.add_row(flags=0, time=2) # 8
        tables.nodes.add_row(flags=0, time=2) # 9
        tables.nodes.add_row(flags=0, time=2) # 10
        tables.nodes.add_row(flags=0, time=2) # 11
        
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
        print(tables.edges)
        sim = tracker.Simulator(
            initial_state=tables
        )
        pairs = sim.P[0].get_num_pairs()
        assert pairs == 14

def make_initial_state(sample_configuration, sequence_length):
    tables = tskit.TableCollection(sequence_length)
    for pop_id, sample_count in enumerate(sample_configuration):
        tables.populations.add_row()
        for _ in range(sample_count):
            tables.nodes.add_row(
                flags=tskit.NODE_IS_SAMPLE, time=0, population=pop_id
            )