import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))

print("Current working directory:", os.getcwd())
print("Python executable:", sys.executable)

from backend import main
from flask import Flask

print("\n--- DIAGNOSIS: START ---")

# Check if model loaded on import
print(f"1. On Import: Model is None? {main.model is None}")

if main.model is None:
    print("   Attempting to load manually...")
    main.load_pretrained_on_startup()
    print(f"   After Manual Load: Model is None? {main.model is None}")

# Mock Request Context
print("\n--- DIAGNOSIS: MOCK REQUEST ---")
app = main.app
with app.test_request_context('/api/predict', method='POST', json={'text': 'test'}):
    try:
        print("2. Calling predict() function directly...")
        response = main.predict()
        print("   Success!")
        print("   Response:", response.get_json())
    except Exception as e:
        print("   FAILED!")
        print("   Error:", str(e))
        import traceback
        traceback.print_exc()

print("\n--- DIAGNOSIS: END ---")
