import inspect
from langchain_ollama import OllamaLLM

# Print the signature of OllamaLLM constructor
print("OllamaLLM signature:", inspect.signature(OllamaLLM.__init__))

# Print the source code of the __init__ method if possible
try:
    print("\nOllamaLLM __init__ source:")
    print(inspect.getsource(OllamaLLM.__init__))
except Exception as e:
    print(f"Could not get source: {e}")

# Print the class MRO (Method Resolution Order)
print("\nOllamaLLM MRO (inheritance):")
print(OllamaLLM.__mro__)

# Try creating an instance with different parameter names
try:
    # Try with base_url parameter
    llm1 = OllamaLLM(base_url="http://localhost:11434", model="llama2")
    print("\nUsing 'model' parameter works")
except Exception as e:
    print(f"\nUsing 'model' parameter failed: {e}")

# Print all available parameters
print("\nAll parameter names in OllamaLLM.__init__:")
try:
    # Get the __init__ method's parameter names
    params = inspect.signature(OllamaLLM.__init__).parameters
    for name, param in params.items():
        print(f"  {name}: {param.default}")
except Exception as e:
    print(f"Error getting parameters: {e}")
