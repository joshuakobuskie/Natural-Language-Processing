import private_api_key
from google import genai
# import timeit

def gemini(prompt="Demonstrate if Gemini API calls work", key=private_api_key.key):
    client = genai.Client(api_key=key)
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response

# Test average execution time
# total_time = timeit.timeit(gemini, number=10)
# print(f"Total time for 10 calls: {total_time:.4f} seconds")
# Total time for 10 calls: 79.8045 seconds
# print(f"Average time per call: {total_time / 10:.4f} seconds")
# Average time per call: 7.9804 seconds

# IMPORTANT: API call time could be a limiting factor if we want to run this many times
# Each call takes about 8 seconds