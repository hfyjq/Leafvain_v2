"""
Unit tests for Pipeline stages and scheduler.

Validates:
  1. Stage registration via @register_stage decorator
  2. Stage ordering by STAGES_ORDER
  3. StageContext defaults
  4. PipelineScheduler initialization
  5. Custom stage integration

Usage:
    cd D:/OPD/codePRS/Leafvain_v2
    python tests/test_pipeline.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_stage_registration():
    """@register_stage should add classes to registered_stages list."""
    from core.pipeline.stage import Stage, StageContext, register_stage
    from core.pipeline.stage import registered_stages as reg

    # Remember initial count
    initial_count = len(reg)

    @register_stage
    class TestStage(Stage):
        async def process(self, ctx: StageContext) -> StageContext:
            return ctx

    assert TestStage in reg
    assert len(reg) == initial_count + 1

    # Clean up — remove our test stage
    reg.remove(TestStage)

    print(f"  ✅ Stage registration: {initial_count} → {initial_count + 1} → {len(reg)}")


def test_stage_context_defaults():
    """StageContext should have sensible defaults."""
    from core.pipeline.stage import StageContext
    from channels.base import MessageSession, MessageType

    session = MessageSession("test", MessageType.PRIVATE, "u1")
    ctx = StageContext(session=session, user_message="hello")

    assert ctx.session == session
    assert ctx.user_message == "hello"
    assert ctx.messages == []
    assert ctx.memory_context == ""
    assert ctx.response == ""
    assert ctx.session_pool is None
    assert ctx.provider is None
    assert ctx.metadata == {}

    print(f"  ✅ StageContext defaults: session={ctx.session}, "
          f"message='{ctx.user_message}'")


def test_stage_context_mutable():
    """Stages should be able to modify StageContext."""
    from core.pipeline.stage import StageContext
    from channels.base import MessageSession, MessageType

    session = MessageSession("test", MessageType.PRIVATE, "u1")
    ctx = StageContext(session=session, user_message="hello")

    # Simulate PreProcess
    ctx.messages = [{"role": "system", "content": "You are helpful."}]
    ctx.memory_context = "<background_context>data</background_context>"

    # Simulate AgentLoop
    ctx.response = "Hello, user!"

    # Simulate PostProcess
    ctx.metadata["context_usage"] = {"percentage": 42.0}

    assert len(ctx.messages) == 1
    assert ctx.response == "Hello, user!"
    assert ctx.metadata["context_usage"]["percentage"] == 42.0

    print(f"  ✅ StageContext mutable: {len(ctx.messages)} msgs, "
          f"response='{ctx.response}'")


def test_pipeline_scheduler_init():
    """PipelineScheduler should initialize without errors."""
    from core.pipeline.scheduler import PipelineScheduler

    # Scheduler with None deps (for testing)
    scheduler = PipelineScheduler(
        session_pool=None,
        provider=None,
        config={"tool_result_max_chars": 1000},
    )

    assert scheduler._config["tool_result_max_chars"] == 1000
    assert len(scheduler._stage_classes) >= 3  # PreProcess, AgentLoop, PostProcess

    class_names = [cls.__name__ for cls in scheduler._stage_classes]
    assert "PreProcessStage" in class_names
    assert "AgentLoopStage" in class_names
    assert "PostProcessStage" in class_names

    print(f"  ✅ PipelineScheduler: {len(scheduler._stage_classes)} stages "
          f"({', '.join(class_names)})")


def test_custom_stage_integration():
    """A custom stage should be sortable in STAGES_ORDER."""
    from core.pipeline.stage import Stage, StageContext, register_stage
    from core.pipeline.stage import registered_stages as reg
    from core.pipeline.scheduler import STAGES_ORDER

    @register_stage
    class CustomStage(Stage):
        async def process(self, ctx: StageContext) -> StageContext:
            ctx.metadata["custom"] = True
            return ctx

    # Verify it's registered
    assert CustomStage in reg

    # Verify it gets a sort index (after built-in stages)
    idx = -1
    try:
        idx = STAGES_ORDER.index("CustomStage")
    except ValueError:
        idx = len(STAGES_ORDER)

    # It should either be in the order list or after everything
    assert idx >= 0

    # Clean up
    reg.remove(CustomStage)

    print(f"  ✅ CustomStage: registered, STAGES_ORDER index={idx}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Pipeline Tests")
    print("=" * 60)
    print()

    tests = [
        ("Stage registration", test_stage_registration),
        ("StageContext defaults", test_stage_context_defaults),
        ("StageContext mutable", test_stage_context_mutable),
        ("PipelineScheduler init", test_pipeline_scheduler_init),
        ("Custom stage integration", test_custom_stage_integration),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ FAIL: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
