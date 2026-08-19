import os
import pytest
from utils import *


# ref: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test
@pytest.fixture(autouse=True)
def stop_server_after_each_test():
    # do nothing before each test
    yield
    # stop all servers after each test
    instances = set(
        server_instances
    )  # copy the set to prevent 'Set changed size during iteration'
    for server in instances:
        server.stop()


@pytest.fixture(scope="session", autouse=True)
def load_server_presets():
    # this will be run once per test session, before any tests
    # SKIP_PRESET_LOAD lets a run reuse an already-populated model cache without network access
    if os.environ.get("SKIP_PRESET_LOAD"):
        return
    ServerPreset.load_all()
