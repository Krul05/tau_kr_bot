from flask import Flask

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "OK", 200

@app.route("/health", methods=["GET"])
def health():
    return {"ok": True}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)