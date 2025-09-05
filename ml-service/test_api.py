import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "age": 22,
    "gender": "male",
    "phq1": 1,
    "phq2": 2,
    "phq3": 0,
    "phq4": 1,
    "phq5": 2,
    "phq6": 0,
    "phq7": 1,
    "phq8": 2,
    "phq9": 1
}
res = requests.post(url, json=payload)
print(res.json())
