import pytest
import time
from infer_facade import infer_best_class_and_keyword_ig, infer_best_class_and_keyword_attention

# Unified test cases (same for IG and Attention)
base_test_cases = [
    ("Right Hip hurts of thigh", "hrt", "hip"),           # IG: hip, Attention: thigh
    ("Ear pain with hearing loss", "ear", "hearing"),     # IG: hearing, Attention: ear
    ("Blurred vision and eye irritation", "EYE", "eye"),  # Both agree on eye
    ("Chest tightness and wheezing", "hrt", "chest"),     # Both often get chest
    ("Pelvic pain with vaginal discharge", "gyn", "pelvic"),  # IG: pelvic, Attention: discharge
    ("Shortness of breath and irregular heartbeat", "hrt", "breath"),  # Both struggle with class
    ("Dizziness and ringing in ears", "ear", "ears"),     # Both agree on ears
    ("Chronic cough with fatigue", "WTC", "cough"),       # Both agree on cough
    ("Thigh stiffness with hip discomfort", "hrt", "thigh"),  # IG: hip, Attention: thigh
    ("Eye redness and sensitivity to light", "EYE", "redness"),  # IG: eye, Attention: eye
]

# Expand to exactly 100 test cases
def expand_test_cases(base_cases, target_count=20):
    expanded = []
    for i in range(1, 3):  # 10 variations per base case
        for note, exp_class, exp_keyword in base_cases:
            expanded.append((f"{note} variation {i}", exp_class, exp_keyword))
    #assert len(expanded) >= target_count, f"Only {len(expanded)} cases, need {target_count}"
    return expanded[:target_count]  # Trim to exactly 100

test_cases = expand_test_cases(base_test_cases, 100)

# Track results
ig_results = {"passes": 0, "total": 0, "times": []}
attention_results = {"passes": 0, "total": 0, "times": []}

def run_ig_tests():
    """Run IG tests and update results."""
    global ig_results
    for note, expected_class, expected_keyword in test_cases:
        ig_results["total"] += 1
        
        start_time = time.time()
        best_class, best_keyword = infer_best_class_and_keyword_ig(note)
        elapsed_time = time.time() - start_time
        ig_results["times"].append(elapsed_time)
        
        # Check class and keyword
        class_pass = best_class == expected_class
        keyword_pass = best_keyword in [expected_keyword, expected_keyword[:-1]] if expected_keyword.endswith('s') else best_keyword == expected_keyword
        test_pass = class_pass and keyword_pass
        
        if test_pass:
            ig_results["passes"] += 1
        
        # Print results
        print(f"IG Test: Note='{note}'")
        print(f"  Expected: Class='{expected_class}', Keyword='{expected_keyword}'")
        print(f"  Got:      Class='{best_class}', Keyword='{best_keyword}'")
        print(f"  Time:     {elapsed_time:.4f} seconds")
        print(f"  Result:   {'PASS' if test_pass else 'FAIL'} (Class: {'PASS' if class_pass else 'FAIL'}, Keyword: {'PASS' if keyword_pass else 'FAIL'})")
        print("-" * 80)

def run_attention_tests():
    """Run Attention tests and update results."""
    global attention_results
    for note, expected_class, expected_keyword in test_cases:
        attention_results["total"] += 1
        
        start_time = time.time()
        best_class, best_keyword = infer_best_class_and_keyword_attention(note)
        elapsed_time = time.time() - start_time
        attention_results["times"].append(elapsed_time)
        
        # Check class and keyword
        class_pass = best_class == expected_class
        keyword_pass = best_keyword in [expected_keyword, expected_keyword[:-1]] if expected_keyword.endswith('s') else best_keyword == expected_keyword
        test_pass = class_pass and keyword_pass
        
        if test_pass:
            attention_results["passes"] += 1
        
        # Print results
        print(f"Attention Test: Note='{note}'")
        print(f"  Expected: Class='{expected_class}', Keyword='{expected_keyword}'")
        print(f"  Got:      Class='{best_class}', Keyword='{best_keyword}'")
        print(f"  Time:     {elapsed_time:.4f} seconds")
        print(f"  Result:   {'PASS' if test_pass else 'FAIL'} (Class: {'PASS' if class_pass else 'FAIL'}, Keyword: {'PASS' if keyword_pass else 'FAIL'})")
        print("-" * 80)

def print_summary():
    """Print pass rates and performance summary."""
    ig_pass_rate = (ig_results["passes"] / ig_results["total"] * 100) if ig_results["total"] > 0 else 0
    ig_avg_time = sum(ig_results["times"]) / len(ig_results["times"]) if ig_results["times"] else 0
    attention_pass_rate = (attention_results["passes"] / attention_results["total"] * 100) if attention_results["total"] > 0 else 0
    attention_avg_time = sum(attention_results["times"]) / len(attention_results["times"]) if attention_results["times"] else 0
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Integrated Gradients (IG):")
    print(f"  Total Tests: {ig_results['total']}")
    print(f"  Passes: {ig_results['passes']}")
    print(f"  Pass Rate: {ig_pass_rate:.2f}%")
    print(f"  Average Time: {ig_avg_time:.4f} seconds")
    print(f"Attention:")
    print(f"  Total Tests: {attention_results['total']}")
    print(f"  Passes: {attention_results['passes']}")
    print(f"  Pass Rate: {attention_pass_rate:.2f}%")
    print(f"  Average Time: {attention_avg_time:.4f} seconds")
    print("=" * 80)

# Pytest tests
@pytest.mark.parametrize("note, expected_class, expected_keyword", test_cases)
def test_infer_ig(note, expected_class, expected_keyword):
    """Pytest wrapper for IG tests."""
    start_time = time.time()
    best_class, best_keyword = infer_best_class_and_keyword_ig(note)
    elapsed_time = time.time() - start_time
    
    class_pass = best_class == expected_class
    keyword_pass = best_keyword in [expected_keyword, expected_keyword[:-1]] if expected_keyword.endswith('s') else best_keyword == expected_keyword
    assert class_pass, f"IG: For note '{note}', expected class '{expected_class}' but got '{best_class}'"
    assert keyword_pass, f"IG: For note '{note}', expected keyword '{expected_keyword}' but got '{best_keyword}'"

@pytest.mark.parametrize("note, expected_class, expected_keyword", test_cases)
def test_infer_attention(note, expected_class, expected_keyword):
    """Pytest wrapper for Attention tests."""
    start_time = time.time()
    best_class, best_keyword = infer_best_class_and_keyword_attention(note)
    elapsed_time = time.time() - start_time
    
    class_pass = best_class == expected_class
    keyword_pass = best_keyword in [expected_keyword, expected_keyword[:-1]] if expected_keyword.endswith('s') else best_keyword == expected_keyword
    assert class_pass, f"Attention: For note '{note}', expected class '{expected_class}' but got '{best_class}'"
    assert keyword_pass, f"Attention: For note '{note}', expected keyword '{expected_keyword}' but got '{best_keyword}'"

if __name__ == "__main__":
    
    # Then run Attention tests
    print("\nRunning Attention Tests...")
    print("=" * 80)
    run_attention_tests()
    

    # Run IG tests first
    print("Running Integrated Gradients Tests...")
    print("=" * 80)
    run_ig_tests()
    
    # Print summary
    print_summary()

# Ensure summary is printed after pytest runs
def pytest_sessionfinish(session, exitstatus):
    print_summary()