"""
Iterative chunk-aggregation summarizer.

NEVER feeds the full document text to the LLM.  Instead:
  Round 1: batch chunks into groups → group summary.
  Round 2+: merge adjacent summaries → intermediate summary.
  Repeat until only 1 summary remains → final output.

For N chunks, the maximum LLM input in any single call is:
  - Round 1: chunks_per_batch * max_chars (adaptive, max ~7500 chars)
  - Merge rounds: merge_batch_size * intermediate_summary_chars (≤ 3×500=1500)

This is orders of magnitude faster than per-chunk summarization for
large documents (47 chunks: 17 calls vs 72 calls originally).
"""

from typing import Dict, List


def iterative_summarize(
    chunks: List[Dict],
    llm_client,  # Provider client with .chat() method
    summary_prompt: str = "You are a document summarizer. Summarize the given text concisely and accurately.",
    chunk_summary_max_chars: int = 200,
    merge_batch_size: int = 3,
) -> str:
    """
    Generate a full-document summary via iterative aggregation.

    Args:
        chunks: List of chunk dicts, each with a ``"text"`` key.
        llm_client: Provider client instance with a ``chat(messages)`` method.
        summary_prompt: System prompt for the summarization LLM calls.
        chunk_summary_max_chars: Target max chars for round-1 per-batch summaries.
        merge_batch_size: Number of adjacent summaries to merge per round.

    Returns:
        Final summary string.

    **Hard constraint**: at no point is the full text of all chunks
    concatenated and sent to the LLM in a single request.
    """
    if not chunks:
        return "(no content to summarize)"

    # ------------------------------------------------------------------
    # Adaptive batch sizing for Round 1
    #   ≤10 chunks → 1 chunk/batch  (original per-chunk behavior)
    #   11-30     → 3 chunks/batch   (~4500 chars per LLM call)
    #   31+       → 5 chunks/batch   (~7500 chars per LLM call)
    # This keeps Round-1 LLM calls bounded while never feeding
    # the full document to the model.
    # ------------------------------------------------------------------
    total = len(chunks)
    if total <= 10:
        round1_batch = 1
    elif total <= 30:
        round1_batch = 3
    else:
        round1_batch = 5

    # Round 1: batch chunks, summarize each batch
    batch_count = (total + round1_batch - 1) // round1_batch
    print(f"[summarizer] Round 1: {total} chunks → {batch_count} batches "
          f"({round1_batch} chunks/batch)")

    summaries: List[str] = []
    for i in range(0, total, round1_batch):
        batch = chunks[i : i + round1_batch]
        combined = "\n\n".join(c["text"] for c in batch)
        # Scale max_chars with batch size — more text needs longer summary
        batch_max_chars = chunk_summary_max_chars * min(round1_batch, 3)
        summary = _summarize_single(
            llm_client,
            system_prompt=summary_prompt,
            text=combined,
            max_chars=batch_max_chars,
        )
        summaries.append(summary)
        print(f"[summarizer]   batch {len(summaries)}/{batch_count} done")

    # ------------------------------------------------------------------
    # Round 2+: iterative merge
    # ------------------------------------------------------------------
    round_num = 2
    while len(summaries) > 1:
        group_count = (len(summaries) + merge_batch_size - 1) // merge_batch_size
        print(f"[summarizer] Round {round_num}: {len(summaries)} summaries "
              f"→ {group_count} groups (merge {merge_batch_size} at a time)")

        next_summaries: List[str] = []
        for i in range(0, len(summaries), merge_batch_size):
            group = summaries[i : i + merge_batch_size]
            if len(group) == 1:
                next_summaries.append(group[0])
            else:
                merged = _merge_summaries(
                    llm_client,
                    system_prompt=summary_prompt,
                    summaries=group,
                )
                next_summaries.append(merged)
        summaries = next_summaries
        round_num += 1

    print(f"[summarizer] Done — {round_num - 1} rounds total")
    return summaries[0]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _summarize_single(
    client,
    system_prompt: str,
    text: str,
    max_chars: int,
) -> str:
    """Ask the LLM to summarize a batch of chunk text."""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Summarize the following text in under {max_chars} characters. "
                f"Focus on key facts, entities, and conclusions:\n\n{text}"
            ),
        },
    ]
    response = client.chat(messages)
    return (response.content or "").strip()


def _merge_summaries(
    client,
    system_prompt: str,
    summaries: List[str],
) -> str:
    """Merge multiple adjacent summaries into one coherent summary."""
    combined = "\n\n---\n\n".join(summaries)
    messages = [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n"
                "Merge the following summaries into one coherent, "
                "non-redundant summary. Eliminate duplicate information."
            ),
        },
        {"role": "user", "content": combined},
    ]
    response = client.chat(messages)
    return (response.content or "").strip()
