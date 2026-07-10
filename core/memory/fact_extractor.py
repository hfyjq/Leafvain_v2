"""
Fact extraction from conversations (Mem0-style).

After each conversation turn, an LLM is asked to extract structured
facts across 7 categories.  New facts are deduplicated against the
existing Chroma vector store before being written.
"""

import json
from typing import Dict, List, Optional

# ------------------------------------------------------------------
# Mem0-style fact extraction prompt — covers 7 information categories
# ------------------------------------------------------------------

FACT_EXTRACTION_PROMPT = """You are a precise fact extraction system. Your job is to extract specific, concrete facts from the conversation.

## Categories
For each fact, classify into exactly one of these categories:
- **personal**: User's personal info (name, role, location, age range, etc.)
- **preference**: Likes, dislikes, habits, preferred tools, languages, workflows
- **project**: Ongoing work, project names, goals, architecture decisions
- **constraint**: Hard requirements, things that must NOT be done, limitations
- **decision**: Choices the user has made or agreed to (technical or otherwise)
- **knowledge**: Technical facts, code snippets, solutions the user shared
- **relationship**: Connections between people, projects, or systems

## Rules
1. Only extract NEW facts not already known.
2. Each fact must be a single, self-contained sentence.
3. Do NOT include opinions, interpretations, or summaries.
4. If the conversation contains no new factual information, return an empty list.
5. Output MUST be valid JSON.

## Output Format
{"facts": [{"category": "...", "fact": "...", "confidence": 0.9}]}

If no new facts are present, return:
{"facts": []}"""


class FactExtractor:
    """
    Extracts structured facts from conversations via LLM, then
    deduplicates against existing facts in the long-term store before
    writing.
    """

    # Confidence below which facts are discarded
    MIN_CONFIDENCE = 0.6

    # Chroma cosine-distance threshold for "this is the same fact"
    DEDUP_DISTANCE = 0.15  # lower distance = more similar

    def __init__(
        self,
        long_term_memory,  # LongTermMemory
        semantic_memory,   # SemanticMemory
        provider,          # BaseProvider for LLM calls
    ) -> None:
        self._long_term = long_term_memory
        self._semantic = semantic_memory
        self._provider = provider
        self._turn_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_and_store(
        self,
        messages: List[Dict],
    ) -> List[Dict]:
        """
        Extract facts from the recent conversation, deduplicate, and
        store in both the vector store and the semantic profile.

        Args:
            messages: The conversation messages to analyze (last few turns).

        Returns:
            List of newly stored fact dicts.
        """
        self._turn_counter += 1

        # 1. Build transcript from recent user + assistant messages
        transcript = _build_transcript(messages, last_n=6)

        # 2. Ask LLM to extract facts
        raw_facts = self._call_extraction_llm(transcript)
        if not raw_facts:
            return []

        # 3. Deduplicate against existing facts in Chroma
        new_facts: List[Dict] = []
        for fact in raw_facts:
            fact_text = fact.get("fact", "")
            category = fact.get("category", "general")
            confidence = fact.get("confidence", 1.0)

            if not fact_text.strip():
                continue
            if confidence < self.MIN_CONFIDENCE:
                continue

            # Search for similar existing facts
            existing = self._long_term.search_facts(fact_text, k=1)
            if existing and existing[0].get("distance", 1.0) < self.DEDUP_DISTANCE:
                # Replace the old fact with the updated one
                old_id = existing[0].get("id", "")
                if old_id:
                    self._long_term.delete_fact(old_id)
                # Fall through to re-add

            new_facts.append(fact)

        # 4. Store new/updated facts in Chroma
        if new_facts:
            self._long_term.add_facts(new_facts)

        # 5. Sync to semantic profile (KV store)
        for fact in new_facts:
            category = fact.get("category", "general")
            fact_text = fact.get("fact", "")
            confidence = fact.get("confidence", 1.0)

            # For preference/constraint/decision categories, also
            # write to the semantic profile as KV pairs
            if category in ("preference", "constraint", "decision", "personal"):
                # Derive a key from the first ~40 chars of the fact
                key = _fact_to_key(fact_text)
                self._semantic.upsert(
                    key=key,
                    value=fact_text,
                    category=category,
                    confidence=confidence,
                    source_turn=self._turn_counter,
                )

        return new_facts

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_extraction_llm(self, transcript: str) -> List[Dict]:
        """Call the LLM to extract facts from a transcript."""
        try:
            response = self._provider.chat([
                {"role": "system", "content": FACT_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Extract new facts from this conversation:\n\n{transcript}"
                    ),
                },
            ])
            content = (response.content or "").strip()
            return _parse_fact_json(content)
        except Exception:
            return []


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_transcript(messages: List[Dict], last_n: int = 6) -> str:
    """Build a compact transcript from the last N user+assistant messages."""
    relevant = [
        m for m in messages[-last_n * 2 :]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    lines: List[str] = []
    for m in relevant[-last_n:]:
        role = "User" if m["role"] == "user" else "Assistant"
        content = m["content"]
        if len(content) > 600:
            content = content[:600] + "..."
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def _parse_fact_json(content: str) -> List[Dict]:
    """Parse the LLM's JSON response into a list of fact dicts."""
    # Strip markdown code fences if present
    text = content.strip()
    if text.startswith("```"):
        # Remove opening fence
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON object anywhere in the response
        import re
        match = re.search(r'\{[^{}]*"facts"\s*:\s*\[.*?\]\s*[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(data, dict):
        return []
    facts = data.get("facts", [])
    if not isinstance(facts, list):
        return []

    return [
        {
            "category": f.get("category", "general"),
            "fact": f.get("fact", ""),
            "confidence": float(f.get("confidence", 1.0)),
        }
        for f in facts
        if isinstance(f, dict) and f.get("fact")
    ]


def _fact_to_key(fact_text: str) -> str:
    """Derive a short key string from a fact for KV storage."""
    # Take first ~40 chars, lowercase, replace spaces with underscores
    key = fact_text[:40].strip().lower()
    # Remove special chars
    key = "".join(c if c.isalnum() or c == "_" else "_" for c in key)
    key = key.strip("_")
    return key if key else "fact_" + fact_text[:20].replace(" ", "_")
