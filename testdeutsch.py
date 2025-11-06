
'''This just makes a requests and prints the resulting json'''

import requests


foo = {"key1": "value1", "key2": "value2"}
req = requests.post(url="http://127.0.0.1:5000/deutsch", json=foo)
print(req.json())
