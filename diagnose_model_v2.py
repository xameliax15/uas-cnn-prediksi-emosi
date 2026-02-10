import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))

print("Current working directory:", os.getcwd())
try:
    from backend import main
except ImportError:
    # Try alternate path if running from root
    sys.path.append(os.getcwd())
    from backend import main

print("\n--- DIAGNOSIS: START (V2) ---")

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
        response_tuple = main.predict()
        
        # Handle tuple return (response, status) or just response
        if isinstance(response_tuple, tuple):
            response, status = response_tuple
        else:
            response = response_tuple
            status = 200
            
        print(f"   Status Code: {status}")
        
        if hasattr(response, 'get_json'):
            data = response.get_json()
            print("   Response JSON:", data)
        else:
            print("   Response Object:", response)
            
    except Exception as e:
        print("   FAILED with Exception!")
        print("   Error:", str(e))
        import traceback
        traceback.print_exc()

print("\n--- DIAGNOSIS: END ---")
