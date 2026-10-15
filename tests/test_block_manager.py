from minivllm.config import SamplingParams
from minivllm.core.block_manager import BlockManager
from minivllm.core.sequence import Sequence


def seq(tokens, rid="r"):
    return Sequence(rid, list(tokens), SamplingParams())


def test_allocate_and_free_roundtrip():
    bm = BlockManager(num_blocks=9, block_size=4)   # 8 usable (block 0 reserved)
    s = seq(range(10))
    assert bm.can_allocate(s)
    bm.allocate(s)
    assert len(s.block_table) == 3 and 0 not in s.block_table
    assert bm.num_free_blocks == 5
    bm.free(s)
    assert bm.num_free_blocks == 8 and s.block_table == []


def test_slot_mapping_uses_physical_blocks():
    bm = BlockManager(num_blocks=16, block_size=4)
    s = seq(range(6))
    bm.allocate(s)
    b0, b1 = s.block_table
    assert bm.slot_mapping(s, 0, 6) == [b0 * 4 + i for i in range(4)] + [b1 * 4, b1 * 4 + 1]


def test_append_slots_grows_block_table_lazily():
    bm = BlockManager(num_blocks=16, block_size=4)
    s = seq(range(4))
    bm.allocate(s)
    assert len(s.block_table) == 1
    assert bm.can_append_slots(s, 1)
    bm.append_slots(s, 1)
    assert len(s.block_table) == 2


def test_prefix_cache_hit_shares_blocks_and_skips_prefill():
    bm = BlockManager(num_blocks=32, block_size=4)
    a = seq(list(range(100, 112)), "a")       # 3 full blocks
    bm.allocate(a)
    a.num_computed_tokens = 12                # pretend the KV was computed
    bm.commit_full_blocks(a)
    b = seq(list(range(100, 108)) + [7, 8, 9], "b")   # shares the first 2 blocks exactly
    bm.allocate(b)
    assert b.block_table[:2] == a.block_table[:2]
    assert b.block_table[2] != a.block_table[2]
    assert b.num_cached_tokens == 8 and b.num_computed_tokens == 8
    assert bm.blocks[a.block_table[0]].ref_count == 2
    assert bm.stats.prefix_hits == 2


def test_prefix_cache_needs_exact_prefix_chain():
    bm = BlockManager(num_blocks=32, block_size=4)
    a = seq(list(range(8)), "a")
    bm.allocate(a)
    a.num_computed_tokens = 8
    bm.commit_full_blocks(a)
    # same second block content, different first block -> no hit (chained hash)
    b = seq([99, 99, 99, 99] + list(range(4, 8)), "b")
    bm.allocate(b)
    assert b.num_cached_tokens == 0


def test_only_computed_blocks_are_committed():
    bm = BlockManager(num_blocks=32, block_size=4)
    a = seq(list(range(8)), "a")
    bm.allocate(a)
    a.num_computed_tokens = 5
    bm.commit_full_blocks(a)
    assert len(bm.cached) == 1


def test_eviction_lru_and_reuse():
    bm = BlockManager(num_blocks=5, block_size=2)    # 4 usable
    a = seq([1, 2, 3, 4], "a")
    bm.allocate(a)
    a.num_computed_tokens = 4
    bm.commit_full_blocks(a)
    bm.free(a)                                        # both blocks now evictable, still cached
    assert bm.stats.num_evictable == 2 and bm.num_free_blocks == 4
    c = seq([1, 2, 9, 9], "c")                        # hits block [1,2]
    bm.allocate(c)
    assert c.num_cached_tokens == 2
    assert bm.num_free_blocks == 2                    # 1 free + 1 evictable ([3,4])
    assert not bm.can_allocate(seq([5, 6, 7, 8, 9, 10], "too-big"))
    d = seq([5, 6, 7, 8], "d")                        # needs 2 blocks: 1 free + evict [3,4]
    bm.allocate(d)
    assert bm.stats.evictions == 1
    assert bm.num_free_blocks == 0
    bm.free(c)
    bm.free(d)
    assert bm.num_free_blocks == 4 and bm.stats.num_evictable == 1   # [1,2] still cached


def test_cannot_allocate_when_full():
    bm = BlockManager(num_blocks=3, block_size=4)     # 2 usable
    a = seq(range(8), "a")
    bm.allocate(a)
    assert not bm.can_allocate(seq(range(4), "b"))
    assert not bm.can_append_slots(a, 1)
