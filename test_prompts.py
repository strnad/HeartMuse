#!/usr/bin/env python3
"""Test script for the new modular prompt template system.

This script verifies that prompts are correctly built for different
field combinations without actually calling the LLM.
"""

from prompt_templates import PromptBuilder


def test_field_combination(fields, context, description):
    """Test a specific field combination."""
    print(f"\n{'=' * 70}")
    print(f"TEST: {description}")
    print(f"Fields: {fields}")
    print(f"{'=' * 70}")

    system_prompt = PromptBuilder.build_system_prompt(fields)
    user_prompt = PromptBuilder.build_user_prompt(fields, context)

    print("\n--- SYSTEM PROMPT (first 300 chars) ---")
    print(system_prompt[:300] + "...")

    print("\n--- USER PROMPT (first 300 chars) ---")
    print(user_prompt[:300] + "...")

    # Verify markers are present
    for field in fields:
        marker = f"==={field.upper()}==="
        if marker in system_prompt:
            print(f"✓ Marker {marker} found in system prompt")
        else:
            print(f"✗ Marker {marker} NOT found in system prompt")

    return True


def main():
    """Run all test cases."""
    print("Testing Modular Prompt Template System")
    print("=" * 70)

    # Test 1: Title only
    test_field_combination(
        fields=["title"],
        context={"description": "A happy pop song", "title": "", "lyrics": "", "tags": ""},
        description="Title only generation"
    )

    # Test 2: Description only
    test_field_combination(
        fields=["description"],
        context={"description": "", "title": "Summer Breeze", "lyrics": "", "tags": "pop,happy,piano"},
        description="Description only generation (from title and tags)"
    )

    # Test 3: Lyrics only
    test_field_combination(
        fields=["lyrics"],
        context={"description": "A melancholic ballad", "title": "Lost in Time", "lyrics": "", "tags": ""},
        description="Lyrics only generation"
    )

    # Test 4: Tags only
    test_field_combination(
        fields=["tags"],
        context={"description": "", "title": "Electric Dreams", "lyrics": "[verse]\nNeon lights...", "tags": ""},
        description="Tags only generation"
    )

    # Test 5: All fields (current default use case)
    test_field_combination(
        fields=["title", "lyrics", "tags"],
        context={"description": "A romantic jazz ballad with saxophone", "title": "", "lyrics": "", "tags": ""},
        description="All fields (title, lyrics, tags) - classic use case"
    )

    # Test 6: Description + Title + Lyrics + Tags (full generation)
    test_field_combination(
        fields=["description", "title", "lyrics", "tags"],
        context={"description": "", "title": "", "lyrics": "", "tags": ""},
        description="Full generation from scratch"
    )

    # Test 7: Partial combination (Title + Lyrics)
    test_field_combination(
        fields=["title", "lyrics"],
        context={"description": "An upbeat rock song", "title": "", "lyrics": "", "tags": "rock,electric guitar,energetic"},
        description="Partial generation (title + lyrics only)"
    )

    # Test 8: Regeneration with existing content
    test_field_combination(
        fields=["lyrics"],
        context={
            "description": "A folk song about nature",
            "title": "Whispering Pines",
            "lyrics": "[verse]\nTrees are swaying...",
            "tags": "folk,acoustic,calm"
        },
        description="Regenerate lyrics with existing title and tags"
    )

    print(f"\n{'=' * 70}")
    print("✓ All test cases completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
