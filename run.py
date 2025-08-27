import requests

url = "http://127.0.0.1:5000/predict"
data = {"review": "The movie was boring and a complete waste of time."}

response = requests.post(url, json=data)
print(response.json())
