
'''Test the classical implementation of the solver for the Deutsch problem'''

import requests


foo = [True, True]
req = requests.post(url="http://127.0.0.1:5000/deutsch-classical", json=foo)
assert req.json()['answer'] == 'constant'

foo = [False, False]
req = requests.post(url="http://127.0.0.1:5000/deutsch-classical", json=foo)
assert req.json()['answer'] == 'constant'

foo = [True, False]
req = requests.post(url="http://127.0.0.1:5000/deutsch-classical", json=foo)
assert req.json()['answer'] == 'balanced'

foo = [False, True]
req = requests.post(url="http://127.0.0.1:5000/deutsch-classical", json=foo)
assert req.json()['answer'] == 'balanced'
