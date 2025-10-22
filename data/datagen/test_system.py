#!/usr/bin/env python3
"""
Test script for the Cognitive Action Data Generation System
Run this to verify everything is working correctly
"""

import sys
import traceback
from variable_pools import *
from prompt_templates import *
from data_generator import *

def test_variable_pools():
    """Test that variable pools are loaded correctly"""
    print("Testing variable pools...")

    # Test cognitive actions
    assert len(COGNITIVE_ACTIONS) > 0, "No cognitive actions loaded"
    print(f"✅ Loaded {len(COGNITIVE_ACTIONS)} cognitive actions")

    # Test subjects
    assert len(SUBJECTS) > 0, "No subjects loaded"
    print(f"✅ Loaded {len(SUBJECTS)} subjects")

    # Test domains
    assert len(DOMAINS) > 0, "No domains loaded"
    print(f"✅ Loaded {len(DOMAINS)} domains")

    # Test random selection
    selection = get_random_selection()
    assert 'cognitive_action' in selection, "Random selection missing cognitive_action"
    print(f"✅ Random selection works: {selection['cognitive_action']}")

    return True

def test_prompt_templates():
    """Test that prompt templates work correctly"""
    print("\nTesting prompt templates...")

    # Test single action template
    prompt, params = generate_prompt(cognitive_action="reconsidering", template_type="single")
    assert len(prompt) > 0, "Empty prompt generated"
    assert "reconsidering" in prompt.lower(), "Cognitive action not in prompt"
    print(f"✅ Single action prompt generated ({len(prompt)} chars)")

    # Test chain template
    prompt, params = generate_prompt(template_type="chain")
    assert len(prompt) > 0, "Empty chain prompt generated"
    print(f"✅ Chain prompt generated ({len(prompt)} chars)")

    # Test dialogue template
    prompt, params = generate_prompt(template_type="dialogue")
    assert len(prompt) > 0, "Empty dialogue prompt generated"
    print(f"✅ Dialogue prompt generated ({len(prompt)} chars)")

    return True

def test_data_generator():
    """Test the data generator class"""
    print("\nTesting data generator...")

    # Test without Ollama (fallback mode)
    generator = CognitiveDataGenerator(ollama_client=None)

    # Generate a test example
    example = generator.generate_single_example(
        cognitive_action="noticing",
        template_type="single"
    )

    assert example is not None, "Failed to generate example"
    assert example.primary_cognitive_action == "noticing", "Wrong cognitive action"
    assert len(example.text) > 0, "Empty example text"
    print(f"✅ Generated example: '{example.text[:50]}...'")

    # Test statistics
    stats = generator.get_statistics()
    assert stats['total_generated'] == 1, "Wrong generation count"
    print(f"✅ Statistics tracking works")

    return True

def test_scientific_taxonomy_coverage():
    """Test that scientific taxonomies are properly represented"""
    print("\nTesting scientific taxonomy coverage...")

    # Check Bloom's taxonomy actions
    bloom_actions = ["remembering", "understanding", "applying", "analyzing", "evaluating", "creating"]
    for action in bloom_actions:
        assert action in COGNITIVE_ACTIONS, f"Missing Bloom's action: {action}"
    print(f"✅ Bloom's taxonomy actions covered")

    # Check emotional regulation actions
    emotion_actions = ["emotional_reappraisal", "emotion_receiving", "situation_selection", "attentional_deployment"]
    for action in emotion_actions:
        assert action in COGNITIVE_ACTIONS, f"Missing emotion regulation action: {action}"
    print(f"✅ Emotion regulation actions covered")

    # Check metacognitive actions
    meta_actions = ["metacognitive_monitoring", "metacognitive_regulation", "meta_awareness"]
    for action in meta_actions:
        assert action in COGNITIVE_ACTIONS, f"Missing metacognitive action: {action}"
    print(f"✅ Metacognitive actions covered")

    return True

def run_all_tests():
    """Run all tests"""
    print("🧪 Running Cognitive Action Data Generation System Tests\n")

    tests = [
        test_variable_pools,
        test_prompt_templates,
        test_data_generator,
        test_scientific_taxonomy_coverage
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! System is ready for data generation.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)