
import sys
import os
import logging

# Ensure Core is in path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPrediction")

from Core.Prediction.predictive_world import PredictiveWorldModel

def test():
    logger.info("🧪 Testing Real Prediction (Mental Sandbox)...")
    
    model = PredictiveWorldModel()
    
    # Test 1: Valid Code Simulation
    logger.info("\n--- Test 1: Valid Code ---")
    valid_code = """
def hello():
    print("Hello, World!")
    return True
"""
    prediction = model.predict_code_impact(
        file_path="test_valid.py",
        change_description="Adding hello function",
        proposed_content=valid_code
    )
    logger.info(f"Prediction: {prediction.description}")
    logger.info(f"Probability: {prediction.probability:.2f}")
    
    # Test 2: Invalid Code Simulation (Syntax Error)
    logger.info("\n--- Test 2: Invalid Code (Syntax Error) ---")
    invalid_code = """
def broken_function()
    print("Missing colon")
"""
    prediction_fail = model.predict_code_impact(
        file_path="test_invalid.py",
        change_description="Adding broken function",
        proposed_content=invalid_code
    )
    logger.info(f"Prediction: {prediction_fail.description}")
    logger.info(f"Probability: {prediction_fail.probability:.2f}")
    
    # Test 3: Complex Code
    logger.info("\n--- Test 3: Complex Code ---")
    complex_code = "import os\n" * 10 + "class A:\n    pass\n" * 5
    prediction_complex = model.predict_code_impact(
        file_path="test_complex.py",
        change_description="Adding complex structure",
        proposed_content=complex_code
    )
    logger.info(f"Prediction: {prediction_complex.description}")
    logger.info(f"Impact Score: {prediction_complex.impact_score:.2f}")

if __name__ == "__main__":
    test()
