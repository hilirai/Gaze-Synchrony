"""Flask web interface for interactive object selection and gaze annotation.

Provides a simple web UI for users to draw positive/negative strokes
on video frames for SAM2 object tracking initialization.
"""

import os
from flask import Flask, render_template, request, jsonify

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.environ.get("SCRIBBLE_STATIC_DIR", os.path.join(ROOT, "static"))
TEMPLATE_DIR = os.environ.get("SCRIBBLE_TEMPLATES_DIR", os.path.join(ROOT, "templates"))

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/static",
    template_folder=TEMPLATE_DIR,
)

strokes = {"pos": [], "neg": []}
selected_frame = 0

@app.route("/")
def index():
    """Serve the main scribble interface."""
    return render_template("scribble.html")

@app.route("/submit", methods=["POST"])
def submit():
    """Receive stroke data and selected frame from client."""
    global strokes, selected_frame
    data = request.get_json(force=True)
    
    if "strokes" in data:
        strokes = data["strokes"]
        selected_frame = data.get("selectedFrame", 0)
    else:
        strokes = data
        selected_frame = 0
        
    print(f"[Flask] Received strokes and selected frame: {selected_frame}")
    return jsonify(status="ok")

@app.route("/get", methods=["GET"])
def get_strokes():
    """Return current strokes and selected frame to calling script."""
    return jsonify({"strokes": strokes, "selectedFrame": selected_frame})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
