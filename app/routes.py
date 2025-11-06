
from app import app
from flask import request
import Deutsch

@app.route('/')
@app.route('/index')
def index():
    return "Welcom to the quantum solver."

@app.route('/deutsch', methods=['POST'])
def deutsch():
    if not request.is_json:
        return "expected json input"
    data = request.json
    return Deutsch.solve(data)
