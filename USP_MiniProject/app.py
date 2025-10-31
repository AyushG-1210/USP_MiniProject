from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return "Live testing 5"

@app.route('/health')
def health():
    # This is a simple health check endpoint for Phase 4
    return "OK", 200

if __name__ == "__main__":
    # Run on 0.0.0.0 to be accessible outside the container
    app.run(host='0.0.0.0', port=5000)
