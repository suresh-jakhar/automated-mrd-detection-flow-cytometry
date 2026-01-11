"""Simple client example to POST to the /predict endpoint"""
import requests

URL = "http://127.0.0.1:5000/predict"

if __name__ == '__main__':
    payload = {'input': [[0.1, 0.2, 0.3, 0.4]]}
    r = requests.post(URL, json=payload)
    print(r.status_code)
    print(r.text)
