import requests

url = "http://127.0.0.1:8000/api/verify/"
data = {
    "inputType": "text",
    "input": "The Earth orbits the Sun."
}

response = requests.post(url, json=data)
print(response.status_code)
print(response.json())
