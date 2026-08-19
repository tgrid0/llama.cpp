import base64
import json
import os
import re
import tempfile
import shutil
import pytest
import requests
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


def test_disk_cache_survives_parallel_count_change():
    """A cache entry saved under one --parallel (n_stream) layout must not stay
    permanently unusable for servers started later with a different --parallel
    sharing the same --cache-disk dir: the mismatched restore should fall back
    to a clean reprocess and replace the incompatible entry, so a later fresh
    server with the new layout gets a real hit instead of reprocessing forever."""
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_nstream_")
    try:
        server.n_slots = 1
        server.kv_unified = False
        server.cache_disk = cache_dir
        server.cache_disk_size = -1
        server.start()

        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        first_prompt_n = res.body["timings"]["prompt_n"]
        assert first_prompt_n > 0

        server.stop()

        server2 = ServerPreset.tinyllama2()
        server2.n_predict = 4
        server2.temperature = 0.0
        server2.n_slots = 4
        server2.kv_unified = False
        server2.cache_disk = cache_dir
        server2.cache_disk_size = -1
        server2.start()

        # Same prompt, different --parallel: the saved entry is unusable here, so
        # this must fall back to a full reprocess (not hang, not crash).
        res2 = server2.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "cache_prompt": True,
        })
        assert res2.status_code == 200
        assert res2.body["timings"]["prompt_n"] == first_prompt_n, (
            "mismatched-layout entry should not be usable; expected a full reprocess"
        )

        server2.stop()

        # A fresh server with the SAME layout as server2 (no in-memory slot state
        # to fall back on) must now see a real disk-cache hit: server2's reprocess
        # should have replaced the incompatible entry rather than leaving it stuck.
        server3 = ServerPreset.tinyllama2()
        server3.n_predict = 4
        server3.temperature = 0.0
        server3.n_slots = 4
        server3.kv_unified = False
        server3.cache_disk = cache_dir
        server3.cache_disk_size = -1
        server3.start()

        res3 = server3.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT,
            "cache_prompt": True,
        })
        assert res3.status_code == 200
        assert res3.body["timings"]["prompt_n"] < first_prompt_n, (
            f"expected disk cache hit for a fresh server sharing server2's layout: "
            f"prompt_n={res3.body['timings']['prompt_n']} should be < {first_prompt_n}"
        )

        server3.stop()
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


def test_disk_cache_prunes_superseded_prefix_entries():
    """A later turn of the same conversation must prune the earlier turn's
    now-superseded disk-cache entry, instead of accumulating one entry per turn
    forever. Each entry holds a full KV-state snapshot, so for a long-running
    multi-turn conversation (as agentic coding clients produce, resending the
    full growing history every turn) unbounded per-turn accumulation wastes
    disk budget and starves LRU eviction of entries other sessions still need.

    Builds prompts from explicit token-ID arrays (not concatenated text) so the
    prefix relationship between turn 1 and turn 2 is exact, independent of any
    BPE re-tokenization boundary effects.
    """
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_prune_")
    try:
        server.cache_disk = cache_dir
        server.cache_disk_size = -1
        server.start()

        def index_entries():
            with open(os.path.join(cache_dir, "index.json")) as f:
                return json.load(f)["entries"]

        long_prompt_tokens = server.make_request("POST", "/tokenize", data={
            "content": LONG_PROMPT,
        }).body["tokens"]

        # Turn 1: fill slot 0, remember the exact generated continuation tokens.
        res1 = server.make_request("POST", "/completion", data={
            "prompt": long_prompt_tokens,
            "id_slot": 0,
            "cache_prompt": True,
            "return_tokens": True,
        })
        assert res1.status_code == 200
        gen1_tokens = res1.body["tokens"]
        assert len(gen1_tokens) > 0

        # Trigger idle-slot save+clear of slot 0 (saves entry #1) by launching a
        # task on slot 1 (cache_idle_slots saves every other non-processing slot).
        server.make_request("POST", "/completion", data={
            "prompt": "Hello",
            "id_slot": 1,
            "cache_prompt": True,
        })

        entries_after_turn1 = index_entries()
        assert len(entries_after_turn1) == 1, entries_after_turn1
        hash1 = next(iter(entries_after_turn1))

        # Turn 2: echo turn 1's full output (prompt + its own generated
        # continuation) plus new content, exactly what a chat client resending
        # conversation history does. No id_slot: slot 0 is idle/cleared, so it
        # gets picked via LRU, which is what makes get_available_slot() attempt
        # the disk-cache restore in the first place.
        extra_tokens = server.make_request("POST", "/tokenize", data={
            "content": " And what happened after that?",
        }).body["tokens"]
        prompt2_tokens = long_prompt_tokens + gen1_tokens + extra_tokens

        res2 = server.make_request("POST", "/completion", data={
            "prompt": prompt2_tokens,
            "cache_prompt": True,
            "return_tokens": True,
        })
        assert res2.status_code == 200

        # Trigger idle-slot save+clear again (saves entry #2, and should prune #1).
        server.make_request("POST", "/completion", data={
            "prompt": "Hello again",
            "id_slot": 1,
            "cache_prompt": True,
        })

        # Slot 1's own "Hello"/"Hello again" prompts get idle-saved as an
        # unrelated, unrelated-lineage side effect of cache_idle_slots (any task
        # launch saves *every* other idle slot, not just the conversation under
        # test), so don't assert a total entry count — assert on the specific
        # lineage instead: turn 1's entry must be gone, and exactly one entry
        # covering the full turn-2 sequence (prompt2 + its own continuation)
        # must remain.
        entries_after_turn2 = index_entries()
        assert hash1 not in entries_after_turn2, (
            "turn 1's now-superseded entry should have been pruned, not kept "
            f"alongside turn 2's: {[(h, e['n_tokens']) for h, e in entries_after_turn2.items()]}"
        )

        lineage_entries = [h for h, e in entries_after_turn2.items() if e["n_tokens"] >= len(prompt2_tokens)]
        assert len(lineage_entries) == 1, (
            f"expected exactly one surviving entry for the long-prompt lineage, got: "
            f"{[(h, e['n_tokens']) for h, e in entries_after_turn2.items()]}"
        )

        bin_files = {os.path.splitext(f)[0] for f in os.listdir(cache_dir) if f.endswith(".bin")}
        assert bin_files == set(entries_after_turn2.keys()), (
            f".bin files on disk must match the index exactly: files={bin_files}, "
            f"index={set(entries_after_turn2.keys())}"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


IMG_URL_TRUCK = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"
IMG_URL_CAT = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"


def _img_b64(url):
    # multimodal_data expects raw base64 (server-common.cpp), not a data: URI
    response = requests.get(url)
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8")


def _vision_server(cache_dir):
    # <__media__> below is only a media marker if this is set before startup
    # (same pattern as test_vision_api.py / test_slot_save.py).
    os.environ['LLAMA_MEDIA_MARKER'] = '<__media__>'
    s = ServerPreset.tinygemma3()
    s.n_slots = 1
    s.n_predict = 4
    s.temperature = 0.0
    s.cache_disk = cache_dir
    s.cache_disk_size = -1
    # tinygemma3's SWA window dwarfs these short prompts, so without --swa-full
    # every restore forces a checkpoint-freshness reset (no checkpoint ever forms).
    s.swa_full = True
    return s


def test_disk_cache_hit_multimodal_after_restart():
    """Exact hit: same image + same text, after a server restart, should reprocess fewer tokens."""
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_mm_")
    try:
        server_mm = _vision_server(cache_dir)
        server_mm.start()

        req = {
            "prompt": {
                "prompt_string": "What is this: <__media__>\n",
                "multimodal_data": [_img_b64(IMG_URL_TRUCK)],
            },
        }
        res = server_mm.make_request("POST", "/completions", data=req)
        assert res.status_code == 200
        first_prompt_n = res.body["timings"]["prompt_n"]
        assert first_prompt_n > 0

        server_mm.stop()

        server_mm2 = _vision_server(cache_dir)
        server_mm2.start()
        res2 = server_mm2.make_request("POST", "/completions", data=req)
        assert res2.status_code == 200
        assert res2.body["timings"]["prompt_n"] < first_prompt_n, (
            f"expected disk cache hit: prompt_n={res2.body['timings']['prompt_n']} should be < {first_prompt_n}"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_disk_cache_multimodal_distinguishes_different_images():
    """Same surrounding text, different images: must NOT be treated as the same cache entry."""
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_mm_")
    fd, log_path = tempfile.mkstemp(suffix=".log", dir=cache_dir)
    os.close(fd)
    try:
        server_mm = _vision_server(cache_dir)
        server_mm.debug = True
        server_mm.log_path = log_path
        server_mm.start()
        log = LogReader(log_path)

        req_truck = {
            "prompt": {
                "prompt_string": "What is this: <__media__>\n",
                "multimodal_data": [_img_b64(IMG_URL_TRUCK)],
            },
        }
        req_cat = {
            "prompt": {
                "prompt_string": "What is this: <__media__>\n",
                "multimodal_data": [_img_b64(IMG_URL_CAT)],
            },
        }

        res_truck = server_mm.make_request("POST", "/completions", data=req_truck)
        assert res_truck.status_code == 200
        log.drain()

        res_cat = server_mm.make_request("POST", "/completions", data=req_cat)
        assert res_cat.status_code == 200

        # [TAG_PROMPT_LOGITS] forces at least one fresh token even on a full cache
        # hit, so prompt_n > 0 alone would also pass for a wrong (truck-served-as-cat)
        # hit. Confirm via the log this request actually missed the truck's cache
        # entry (exact hash differs by image, and the shared text prefix alone is
        # short of the 50% prefix-match threshold), the same way test 3 (edited-turn
        # prefix hit) confirms a hit.
        log_content = log.drain()
        assert "disk cache: exact miss" in log_content or "disk cache: no prefix match" in log_content, (
            f"expected the cat request to miss the truck's cache entry, got:\n{log_content}"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_disk_cache_prefix_hit_multimodal_edited_turn():
    """Prefix hit: same image, edited trailing text, should still reuse the image's cached prefix."""
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_mm_")
    fd, log_path = tempfile.mkstemp(suffix=".log", dir=cache_dir)
    os.close(fd)
    try:
        server_mm = _vision_server(cache_dir)
        server_mm.start()

        base_req = {
            "prompt": {
                "prompt_string": "What is this: <__media__>\nDescribe it in one word.\n",
                "multimodal_data": [_img_b64(IMG_URL_TRUCK)],
            },
        }
        res = server_mm.make_request("POST", "/completions", data=base_req)
        assert res.status_code == 200

        server_mm.stop()

        server_mm2 = _vision_server(cache_dir)
        server_mm2.debug = True
        server_mm2.log_path = log_path
        server_mm2.start()
        log = LogReader(log_path)
        log.drain()

        edited_req = {
            "prompt": {
                "prompt_string": "What is this: <__media__>\nName the color.\n",
                "multimodal_data": [_img_b64(IMG_URL_TRUCK)],
            },
        }
        res2 = server_mm2.make_request("POST", "/completions", data=edited_req)
        assert res2.status_code == 200
        # the image prefix should be reused; only the edited trailing text is new
        assert res2.body["timings"]["prompt_n"] < res.body["timings"]["prompt_n"] + 5

        # Confirm via the log this was an actual prefix hit, not a coincidentally
        # small cold reprocess (find_best_prefix() logs this line on a match).
        log_content = log.drain()
        assert "disk cache: prefix match" in log_content, (
            f"expected a disk-cache prefix-match log line, got:\n{log_content}"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_disk_cache_multimodal_text_then_image_same_server():
    """Regression: restoring a text-only disk-cache entry must not clear the slot's
    has_mtmd flag on a multimodal-capable server, or the next image on that same
    slot aborts the process (GGML_ASSERT in server_tokens::push_back)."""
    cache_dir = tempfile.mkdtemp(prefix="llama_disk_cache_mm_")
    try:
        server_mm = _vision_server(cache_dir)
        server_mm.start()

        text_req = {"prompt": "Hello there, how are you today?"}
        res1 = server_mm.make_request("POST", "/completions", data=text_req)
        assert res1.status_code == 200

        # n_slots=1: the same slot is reused, saving the first request's state to
        # disk cache and then hitting it (exact or prefix) for this identical prompt
        res2 = server_mm.make_request("POST", "/completions", data=text_req)
        assert res2.status_code == 200

        # slot is reused again for an image request; before the fix, has_mtmd was
        # left false by the text-only restore above and this crashed the server
        img_req = {
            "prompt": {
                "prompt_string": "What is this: <__media__>\n",
                "multimodal_data": [_img_b64(IMG_URL_TRUCK)],
            },
        }
        res3 = server_mm.make_request("POST", "/completions", data=img_req)
        assert res3.status_code == 200
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
