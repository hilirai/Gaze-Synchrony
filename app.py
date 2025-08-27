import os
from flask import Flask, render_template, request, jsonify

# === Resolve shared paths (can be overridden via env) ===
ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.environ.get("SCRIBBLE_STATIC_DIR", os.path.join(ROOT, "static"))
TEMPLATE_DIR = os.environ.get("SCRIBBLE_TEMPLATES_DIR", os.path.join(ROOT, "templates"))

# Ensure folders exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# Serve /static from STATIC_DIR and templates from TEMPLATE_DIR
app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/static",
    template_folder=TEMPLATE_DIR,
)

# In-memory store for strokes
strokes = {"pos": [], "neg": []}

@app.route("/")
def index():
    return render_template("scribble.html")

@app.route("/submit", methods=["POST"])
def submit():
    global strokes
    strokes = request.get_json(force=True)  # {"pos":[[x,y]...], "neg":[[x,y]...]}
    return jsonify(status="ok")

@app.route("/get", methods=["GET"])
def get_strokes():
    return jsonify(strokes)

if __name__ == "__main__":
    # Run the tiny server
    app.run(host="0.0.0.0", port=5000)
