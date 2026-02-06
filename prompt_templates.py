"""Modular prompt template system for text generation.

This module provides a flexible, component-based approach to building prompts
for song metadata generation (title, description, lyrics, tags). Prompts are
assembled from reusable components based on which fields need to be generated.
"""

from typing import List, Dict, Optional


class PromptTemplates:
    """Container for modular prompt template components."""

    # Tier 1: Base components (shared across all prompts)
    BASE_ROLE = "You are an expert songwriter and music producer."

    THINKING_STEPS = {
        "all": """IMPORTANT: Think step by step to ensure all fields are coherent with each other:
1. First, come up with the TITLE — it should capture the core theme/emotion of the song description.
2. Then, write the LYRICS — they must reflect the title and the description. The title should feel like a natural fit for the lyrics.
3. Finally, derive the TAGS — base them on the actual musical qualities of the lyrics and title you just wrote, not just the description.""",

        "partial": """IMPORTANT: When generating fields, ensure they are coherent with any provided context (existing title, lyrics, tags, description).""",
    }

    FORMAT_INSTRUCTIONS = """You MUST output EXACTLY the requested fields in the specified format. Only output fields that are requested. Do not output fields that are not requested."""

    # Tier 2: Field-specific guidelines
    FIELD_GUIDELINES = {
        "title": """For title, keep it 1-5 words, creative and fitting. The title should capture the core theme/emotion.""",

        "description": """For description, write 1-3 concise, engaging sentences that capture the essence, mood, style, and key musical elements. Be specific about genre, instrumentation, and emotional quality.""",

        "lyrics": """For lyrics, use sections like [verse], [chorus], [bridge], [outro], [intro]. Keep lyrics natural and singable.
CRITICAL PRESERVATION RULES for existing lyrics:
1. Text in quotation marks ("text") → MUST be preserved 100% EXACTLY, word-for-word, character-for-character. This is ABSOLUTE.
2. Text without quotation marks → Preserve very closely (keep the same meaning, structure, and most words, but minor variations are acceptable).
3. Only add new sections or extend existing ones. Never completely remove or rewrite the provided text.
IMPORTANT: When writing or editing lyrics, convert any uppercase abbreviations to dotted form (e.g., "AI" -> "A.I.", "FBI" -> "F.B.I.", "IT" -> "I.T.", "USB" -> "U.S.B.") so they are pronounced letter by letter.""",

        "tags": """For tags, use comma-separated MUSIC PRODUCTION tags ONLY. Each tag is a single word or a short multi-word phrase. Use 5-15 tags total. Choose from these categories:
- Genre/subgenre: pop, rock, jazz, classical, electronic, blues, country, funk, reggae, hip-hop, rap, indie, folk, soul, r&b, latin, dance, disco, techno, house, ambient, lo-fi, hard rock, progressive rock, dream pop, synth-pop, bossa nova
- Mood/emotion: happy, sad, romantic, energetic, calm, peaceful, melancholic, upbeat, chill, dark, bright, dreamy, aggressive, gentle, intense, relaxing, cheerful, nostalgic, dramatic, playful
- Instruments: piano, guitar, bass, drum, violin, saxophone, trumpet, flute, organ, synthesizer, acoustic guitar, electric guitar, cello, harp, strings, brass
- Vocal type: male voice, female voice, choir, vocal harmony, rap, duet (MUST match the song description)
- Tempo/style: fast, slow, medium tempo, groovy, driving, flowing, bouncy, acoustic, orchestral, cinematic, atmospheric, minimal, lo-fi, polished
Do NOT include content/story/character/theme tags (e.g. no "love", "heartbreak", "summer" as tags). Focus strictly on how the music SOUNDS, not what it is about.""",
    }

    # Tier 3: Field markers (format specifications)
    FIELD_MARKERS = {
        "title": """===TITLE===
song title here (1-5 words, no quotes)
===END_TITLE===
""",
        "description": """===DESCRIPTION===
description here (1-3 sentences)
===END_DESCRIPTION===
""",
        "lyrics": """===LYRICS===
[verse]
lyrics here

[chorus]
lyrics here
===END_LYRICS===
""",
        "tags": """===TAGS===
comma,separated,tags,here
===END_TAGS===
""",
    }


class PromptBuilder:
    """Builds prompts dynamically based on requested fields."""

    @staticmethod
    def build_system_prompt(fields: List[str]) -> str:
        """Build a system prompt for generating specified fields.

        Args:
            fields: List of field names to generate (e.g., ["title", "lyrics", "tags"])

        Returns:
            Complete system prompt string
        """
        parts = []

        # 1. Base role
        parts.append(PromptTemplates.BASE_ROLE)
        parts.append("")

        # 2. Thinking steps (full if generating multiple fields, simplified if single)
        if len(fields) > 1:
            parts.append(PromptTemplates.THINKING_STEPS["all"])
        else:
            parts.append(PromptTemplates.THINKING_STEPS["partial"])
        parts.append("")

        # 3. Format instructions
        parts.append(PromptTemplates.FORMAT_INSTRUCTIONS)
        parts.append("")

        # 4. Field markers (show format for requested fields)
        for field in fields:
            if field in PromptTemplates.FIELD_MARKERS:
                parts.append(PromptTemplates.FIELD_MARKERS[field])

        # 5. Field-specific guidelines (only for requested fields)
        for field in fields:
            if field in PromptTemplates.FIELD_GUIDELINES:
                parts.append(PromptTemplates.FIELD_GUIDELINES[field])

        return "\n".join(parts)

    @staticmethod
    def build_user_prompt(fields: List[str], context: Dict[str, str],
                         max_length_sec: Optional[int] = None) -> str:
        """Build user prompt with context for field generation.

        Args:
            fields: List of fields to generate
            context: Dict with keys like 'description', 'title', 'lyrics', 'tags' (existing values)
            max_length_sec: Optional duration constraint

        Returns:
            User prompt string
        """
        parts = []

        # Determine if we have any context
        has_context = any(context.get(k, "").strip() for k in ["description", "title", "lyrics", "tags"])

        if has_context:
            # Build context section
            context_parts = []

            # Description (always as context if present and not being generated)
            if context.get("description", "").strip():
                if "description" in fields:
                    context_parts.append(f"Current description draft (to be improved): {context['description']}")
                else:
                    context_parts.append(f"Song description: {context['description']}")

            # Other fields - show as "already set" if not being regenerated
            for field in ["title", "lyrics", "tags"]:
                value = context.get(field, "").strip()
                if not value:
                    continue

                if field in fields:
                    # Being regenerated - show as draft to improve/extend
                    if field == "lyrics":
                        context_parts.append(f"Existing lyrics (MUST be preserved exactly, only extend/complete):\n{value}")
                    else:
                        context_parts.append(f"Current {field} draft (to be improved): {value}")
                else:
                    # Not being regenerated - show as fixed context
                    if field == "lyrics":
                        context_parts.append(f"{field.capitalize()} (already set):\n{value}")
                    else:
                        context_parts.append(f"{field.capitalize()} (already set): {value}")

            # Duration constraint
            if max_length_sec:
                context_parts.append(
                    f"Target song duration: {max_length_sec} seconds. "
                    f"Adjust lyrics length accordingly — shorter songs (under 60s) need only 1-2 short sections, "
                    f"longer songs (over 120s) can have more verses, a bridge, etc."
                )

            context_str = "\n\n".join(context_parts)
            fields_str = ", ".join(fields).upper()

            parts.append(f"Based on the following context, generate these fields: {fields_str}")
            parts.append("")
            parts.append(context_str)
            parts.append("")
            parts.append("Generate ONLY the requested fields using the exact marker format specified.")
        else:
            # No context - generate from scratch
            fields_str = ", ".join(fields).upper()
            parts.append(f"Generate creative and original content for a new song. Create these fields: {fields_str}")
            parts.append("")
            parts.append("Be creative and come up with an interesting, unique song concept. Make sure all fields are coherent with each other.")
            parts.append("")
            parts.append("Generate ONLY the requested fields using the exact marker format specified.")

        return "\n".join(parts)
