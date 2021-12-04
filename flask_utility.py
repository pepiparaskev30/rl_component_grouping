from flask import Flask, render_template, request
import sqlite3
import random


"""
An API that returns the utility value for each deployment
"""
app = Flask(__name__)

@app.route('/utility', methods=['POST', 'GET'])

def utility_value():
    return str(random.uniform(0.1, 0.99))


if __name__ == "__main__":        # on running python app.py
    app.run(port=5003)  