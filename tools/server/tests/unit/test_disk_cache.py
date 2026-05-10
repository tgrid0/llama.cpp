import os
import re
import tempfile
import shutil
import pytest
from utils import *

server = ServerPreset.tinyllama2()

LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.kv_unified = True


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0

    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


def test_disk_cache_hit_after_restart():
    """Disk cache persists across restarts: second run should process fewer tokens than first."""
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_")
    try:
        server.cache_disk = cache_dir
        server.cache_disk_size = -1
        server.start()

        # First request fills slot 0
        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "id_slot": 0,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        first_prompt_n = res.body["timings"]["prompt_n"]
        assert first_prompt_n > 0

        # Trigger slot 0 idle save by starting a task on slot 1
        server.make_request("POST", "/completion", data={
            "prompt": "Hello",
            "id_slot": 1,
            "cache_prompt": True,
        })

        server.stop()

        # Restart with the same cache dir
        server2 = ServerPreset.tinyllama2()
        server2.n_slots = 2
        server2.n_predict = 4
        server2.temperature = 0.0
        server2.kv_unified = True
        server2.cache_disk = cache_dir
        server2.cache_disk_size = -1
        server2.start()

        # Same prompt — KV state should be loaded from disk (fewer tokens processed)
        res2 = server2.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "cache_prompt": True,
        })
        assert res2.status_code == 200
        assert res2.body["timings"]["prompt_n"] < first_prompt_n, (
            f"expected disk cache hit: prompt_n={res2.body['timings']['prompt_n']} should be < {first_prompt_n}"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_disk_cache_idle_slot_clear_and_restore():
    """Idle slot is saved to disk and freed; submitting the same prompt restores it from disk."""
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_")
    fd, log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    try:
        server.cache_disk = cache_dir
        server.cache_disk_size = -1
        server.debug = True
        server.log_path = log_path
        server.start()
        log = LogReader(log_path)

        assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.drain()

        # Fill slot 0 with the long prompt
        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "id_slot": 0,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        original_prompt_n = res.body["timings"]["prompt_n"]

        # Slot 0 is the only occupied slot — should not be saved yet
        assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

        # Starting slot 1 triggers slot 0 idle save to disk + clear
        server.make_request("POST", "/completion", data={
            "prompt": "The quick brown fox",
            "id_slot": 1,
            "cache_prompt": True,
        })
        assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.drain()

        # Re-send LONG_PROMPT — KV should be restored from disk cache
        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        assert res.body["timings"]["prompt_n"] < original_prompt_n, (
            "expected disk cache hit with fewer prompt tokens after idle slot clear"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.unlink(log_path)


def test_disk_cache_prefix_search():
    """Prefix search: a longer prompt partially hits a shorter cached prefix."""
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_prefix_")
    try:
        server.cache_disk = cache_dir
        server.cache_disk_size = -1
        server.start()

        # Fill slot 0 with LONG_PROMPT and record how many tokens it processes
        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "id_slot": 0,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        first_prompt_n = res.body["timings"]["prompt_n"]
        assert first_prompt_n > 0

        # Trigger idle save of slot 0 by running a task on slot 1
        server.make_request("POST", "/completion", data={
            "prompt": "Hello",
            "id_slot": 1,
            "cache_prompt": True,
        })

        # Send LONG_PROMPT + extra text — exact hash won't match, but prefix will.
        # The prefix covers all of LONG_PROMPT, so only the suffix tokens are processed.
        extended_prompt = LONG_PROMPT + " The adventure continued beyond the forest."
        res2 = server.make_request("POST", "/completion", data={
            "prompt": extended_prompt,
            "cache_prompt": True,
        })
        assert res2.status_code == 200

        # With a prefix hit, only the extra suffix tokens need processing — far fewer
        # than the full LONG_PROMPT token count processed on the first run.
        extended_n = res2.body["timings"]["prompt_n"]
        assert extended_n < first_prompt_n, (
            f"expected disk prefix cache hit: prompt_n={extended_n} should be < {first_prompt_n}"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_disk_cache_hit_after_clean_shutdown():
    """Active slot KV state is flushed to disk on shutdown; restarted server loads it."""
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_shutdown_")
    try:
        server.cache_disk = cache_dir
        server.cache_disk_size = -1
        server.start()

        # Send one completion — fills slot 0; do NOT trigger idle save via a second slot
        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "id_slot": 0,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        first_prompt_n = res.body["timings"]["prompt_n"]
        assert first_prompt_n > 0

        # Stop server — shutdown flush should save slot 0 to disk
        server.stop()

        # Verify that at least one .bin file now exists in the cache dir
        bin_files = [f for f in os.listdir(cache_dir) if f.endswith(".bin")]
        assert len(bin_files) > 0, "expected at least one .bin cache file after shutdown"

        # Restart with same cache dir
        server2 = ServerPreset.tinyllama2()
        server2.n_slots = 2
        server2.n_predict = 4
        server2.temperature = 0.0
        server2.kv_unified = True
        server2.cache_disk = cache_dir
        server2.cache_disk_size = -1
        server2.start()

        # Same prompt — should hit disk cache (fewer tokens processed)
        res2 = server2.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "cache_prompt": True,
        })
        assert res2.status_code == 200
        assert res2.body["timings"]["prompt_n"] < first_prompt_n, (
            f"expected disk cache hit after clean shutdown: "
            f"prompt_n={res2.body['timings']['prompt_n']} should be < {first_prompt_n}"
        )

        server2.stop()
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_disk_cache_prefix_hit_token_count_matches_restored_state():
    """Regression test: after a prefix-hit disk-cache restore, prompt.tokens length
    must match the restored memory's pos_max + 1.

    Request A fills slot 0 and generates n_predict (4) tokens beyond the prompt, so
    the slot's KV state ends up at pos_max = n_prompt + 3. Triggering the idle-slot
    save (via a request on slot 1) saves slot 0's full state (prompt + generated
    continuation) to disk and clears slot 0. A follow-up request with an extended
    prompt then misses the exact hash but hits the saved entry via prefix search.
    The entry's token sequence must cover the full saved KV state (prompt +
    continuation), not just the original prompt, otherwise prompt.tokens ends up
    shorter than pos_max + 1.
    """
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_consist_")
    fd, log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    try:
        server.cache_disk = cache_dir
        server.cache_disk_size = -1
        server.debug = True
        server.log_path = log_path
        server.start()
        log = LogReader(log_path)

        # Request A: fills slot 0 and generates n_predict (4) tokens beyond the
        # prompt, so the slot's KV state ends up at pos_max = n_prompt + 3.
        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "id_slot": 0,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        log.drain()

        # Trigger idle-slot save+clear of slot 0 by starting a task on slot 1.
        server.make_request("POST", "/completion", data={
            "prompt": "Hello",
            "id_slot": 1,
            "cache_prompt": True,
        })
        assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.drain()

        # Request B: LONG_PROMPT + extra suffix. Slot 0 is now empty, so it is
        # selected via LRU, missing the exact hash but hitting the saved entry
        # via prefix search (load_by_hash) of the entry saved above.
        extended_prompt = LONG_PROMPT + " The adventure continued beyond the forest."
        res2 = server.make_request("POST", "/completion", data={
            "prompt": extended_prompt,
            "cache_prompt": True,
        })
        assert res2.status_code == 200

        logs = log.drain()
        m = re.search(
            r"disk prefix-hit, restored state; prompt\.tokens=(\d+) \(req=\d+\), checkpoints=\d+, mem pos_min=(-?\d+) pos_max=(-?\d+)",
            logs,
        )
        assert m, f"expected a disk prefix-hit log line, got:\n{logs}"
        prompt_tokens = int(m.group(1))
        pos_max = int(m.group(3))
        assert prompt_tokens == pos_max + 1, (
            f"prompt.tokens={prompt_tokens} should equal pos_max+1={pos_max + 1} "
            "after a prefix-hit disk restore (entry.tokens must cover the full "
            "saved KV state, not just the original prompt)"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.unlink(log_path)
