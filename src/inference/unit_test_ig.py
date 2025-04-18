import pytest
from infer_facade import infer_best_class_and_keyword_ig, infer_best_class_and_keyword_attention

# Test cases: (note, expected_class, expected_keyword)
test_cases = [
    ("Right Hip hurts of thigh", "hrt", "hip"),
    ("Ear pain with hearing loss", "ear", "ear"),
    ("Blurred vision and eye irritation", "EYE", "vision"),
    ("Chest tightness and wheezing", "hrt", "chest"),  # Could also be WTC, but hrt is primary in data
    ("Pelvic pain with vaginal discharge", "gyn", "pelvic"),
    ("Shortness of breath and irregular heartbeat", "hrt", "breath"),
    ("Dizziness and ringing in ears", "ear", "ears"),
    ("Chronic cough with fatigue", "WTC", "cough"),
]

@pytest.mark.parametrize("note, expected_class, expected_keyword", test_cases)
def test_infer_ig(note, expected_class, expected_keyword):
    """
    Test the Integrated Gradients inference function.
    """
    best_class, best_keyword = infer_best_class_and_keyword_ig(note)
    
    # Assertions
    assert best_class == expected_class, (
        f"IG: For note '{note}', expected class '{expected_class}' but got '{best_class}'"
    )
    assert best_keyword == expected_keyword, (
        f"IG: For note '{note}', expected keyword '{expected_keyword}' but got '{best_keyword}'"
    )
    print(f"IG Passed: Note='{note}', Class='{best_class}', Keyword='{best_keyword}'")

@pytest.mark.parametrize("note, expected_class, expected_keyword", test_cases)
def test_infer_attention(note, expected_class, expected_keyword):
    """
    Test the Attention-based inference function.
    """
    best_class, best_keyword = infer_best_class_and_keyword_attention(note)
    
    # Assertions
    assert best_class == expected_class, (
        f"Attention: For note '{note}', expected class '{expected_class}' but got '{best_class}'"
    )
    assert best_keyword == expected_keyword, (
        f"Attention: For note '{note}', expected keyword '{expected_keyword}' but got '{best_keyword}'"
    )
    print(f"Attention Passed: Note='{note}', Class='{best_class}', Keyword='{best_keyword}'")

if __name__ == "__main__":
    # Run tests manually if executed directly
    for note, exp_class, exp_keyword in test_cases:
        # Test IG
        class_ig, keyword_ig = infer_best_class_and_keyword_ig(note)
        print(f"IG Test: Note='{note}', Expected=({exp_class}, {exp_keyword}), Got=({class_ig}, {keyword_ig})")
        
        # Test Attention
        class_att, keyword_att = infer_best_class_and_keyword_attention(note)
        print(f"Attention Test: Note='{note}', Expected=({exp_class}, {exp_keyword}), Got=({class_att}, {keyword_att})")
