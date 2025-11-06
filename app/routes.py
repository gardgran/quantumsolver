
from app import app
from flask import request
import DeutschClassical

@app.route('/')
@app.route('/index')
def index():
    return "Welcom to the quantum solver."

@app.route('/deutsch-classical', methods=['POST'])
def deutsch():
    if not request.is_json:
        return "expected json input"
    data = request.json
    return DeutschClassical.solve(data)
