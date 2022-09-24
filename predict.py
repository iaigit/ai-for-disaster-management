import base64
import time

import requests

start = time.time()

with open("./images/sample_image.jpg", "rb") as f:
    imgstr = base64.b64encode(f.read()).decode("UTF-8")

body = {"session": "UUID", "payload": {"inputs": {"data": imgstr}}}
resp = requests.post("http://127.0.0.1:8000/predict", json=body)
print(resp.json())
end = time.time()

print(end - start)
