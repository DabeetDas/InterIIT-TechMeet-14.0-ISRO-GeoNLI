"""
Script to send POST request to the API endpoint with sample1_query.json content.
"""

import json
import time
import requests

# API endpoint
url = "http://127.0.0.1:5000/api"

# Read the JSON file
with open('sample3_query.json', 'r') as f:
    payload = json.load(f)

# Set headers
headers = {
    'Content-Type': 'application/json'
}

# Send POST request
try:
    start_time = time.time()
    response = requests.post(url, json=payload, headers=headers)
    
    # Check if request was successful
    response.raise_for_status()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(response)
    # Print the response
    print("Status Code:", response.status_code)
    print("\nResponse JSON:")
    print(json.dumps(response.json(), indent=2))
    
except requests.exceptions.RequestException as e:
    print(f"Error occurred: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response Status Code: {e.response.status_code}")
        print(f"Response Text: {e.response.text}")

