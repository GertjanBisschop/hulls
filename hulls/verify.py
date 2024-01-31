import itertools

from hulls import algorithm as alg


class OverlapCounter:
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.overlaps = self._make_segment(0, seq_length, 0)

    def overlaps_at(self, pos):
        assert 0 <= pos < self.seq_length
        curr_interval = self.overlaps
        while curr_interval is not None:
            if curr_interval.left <= pos < curr_interval.right:
                return curr_interval.node
            curr_interval = curr_interval.next
        raise ValueError("Bad overlap count chain")

    def increment_interval(self, left, right):
        """
        Increment the count that spans the interval
        [left, right), creating additional intervals in overlaps
        if necessary.
        """
        curr_interval = self.overlaps
        while left < right:
            if curr_interval.left == left:
                if curr_interval.right <= right:
                    curr_interval.node += 1
                    left = curr_interval.right
                    curr_interval = curr_interval.next
                else:
                    self._split(curr_interval, right)
                    curr_interval.node += 1
                    break
            else:
                if curr_interval.right < left:
                    curr_interval = curr_interval.next
                else:
                    self._split(curr_interval, left)
                    curr_interval = curr_interval.next

    def _split(self, seg, bp):  # noqa: A002
        """
        Split the segment at breakpoint and add in another segment
        from breakpoint to seg.right. Set the original segment's
        right endpoint to breakpoint
        """
        right = self._make_segment(bp, seg.right, seg.node)
        if seg.next is not None:
            seg.next.prev = right
            right.next = seg.next
        right.prev = seg
        seg.next = right
        seg.right = bp

    def _make_segment(self, left, right, count):
        seg = alg.Segment(0)
        seg.left = left
        seg.right = right
        seg.node = count
        return seg

    def get_total(self):
        total = 0
        curr_interval = self.overlap
        while curr_interval is not None:
            total += curr_interval.node
            curr_interval = curr_interval.next
        return total


def verify_hulls(sim):
    pass


def verify_segments(sim):
    for pop in sim.P:
        for label in range(sim.num_labels):
            for head in pop.iter_label(label):
                assert head.prev is None
                prev = head
                u = head.next
                while u is not None:
                    assert prev.next is u
                    assert u.prev is prev
                    assert u.left >= prev.right
                    assert u.label == head.label
                    assert u.population == head.population
                    prev = u
                    u = u.next


def verify_overlaps(sim):
    overlap_counter = OverlapCounter(sim.L)
    subtotal = 0
    for pop in sim.P:
        for label in range(sim.num_labels):
            for u in pop.iter_label(label):
                while u is not None:
                    overlap_counter.increment_interval(u.left, u.right)
                    u = u.next
            subtotal += pop.get_num_pairs(label)
            overlap_counter_total = overlap_counter.get_total()
            assert overlap_counter_total == subtotal

    for pos, count in sim.S.items():
        if pos != sim.L:
            assert count == overlap_counter.overlaps_at(pos)

    assert sim.S[sim.L] == -1
    # Check the ancestry tracking.
    A = bintrees.AVLTree()
    A[0] = 0
    A[sim.L] = -1
    for pop in sim.P:
        for label in range(sim.num_labels):
            for u in pop.iter_label(label):
                while u is not None:
                    if u.left not in A:
                        k = A.floor_key(u.left)
                        A[u.left] = A[k]
                    if u.right not in A:
                        k = A.floor_key(u.right)
                        A[u.right] = A[k]
                    k = u.left
                    while k < u.right:
                        A[k] += 1
                        k = A.succ_key(k)
                    u = u.next
    # Now, defrag A
    j = 0
    k = 0
    while k < sim.L:
        k = A.succ_key(j)
        if A[j] == A[k]:
            del A[k]
        else:
            j = k
    assert list(A.items()) == list(sim.S.items())


def verify_mass_index(sim, label, mass_index, rate_map, compute_left_bound):
    assert mass_index is not None
    total_mass = 0
    alt_total_mass = 0
    for pop_index, pop in enumerate(sim.P):
        for u in pop.iter_label(label):
            assert u.prev is None
            left = compute_left_bound(u)
            while u is not None:
                assert u.population == pop_index
                assert u.left < u.right
                left_bound = compute_left_bound(u)
                s = rate_map.mass_between(left_bound, u.right)
                right = u.right
                index_value = mass_index.get_value(u.index)
                total_mass += index_value
                assert math.isclose(s, index_value, abs_tol=1e-6)
                v = u.next
                if v is not None:
                    assert v.prev == u
                    assert u.right <= v.left
                u = v

            s = rate_map.mass_between(left, right)
            alt_total_mass += s
    assert math.isclose(total_mass, mass_index.get_total(), abs_tol=1e-6)
    assert math.isclose(total_mass, alt_total_mass, abs_tol=1e-6)


def verify(sim):
    """
    Checks that the state of the simulator is consistent.
    """
    sim.verify_segments()
    if sim.model != "fixed_pedigree":
        # The fixed_pedigree model doesn't maintain a bunch of stuff.
        # It would probably be simpler if it did.
        sim.verify_overlaps()
        for label in range(sim.num_labels):
            if sim.recomb_mass_index is None:
                assert sim.recomb_map.total_mass == 0
            else:
                sim.verify_mass_index(
                    label,
                    sim.recomb_mass_index[label],
                    sim.recomb_map,
                    sim.get_recomb_left_bound,
                )

            if sim.gc_mass_index is None:
                assert sim.gc_map.total_mass == 0
            else:
                sim.verify_mass_index(
                    label,
                    sim.gc_mass_index[label],
                    sim.gc_map,
                    sim.get_gc_left_bound,
                )
