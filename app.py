import os
from io import BytesIO

from flask import Flask, jsonify, render_template, request
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

    model_name = os.getenv("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.post("/caption")
    def caption():
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        file_storage = request.files["image"]
        if not file_storage.filename:
            return jsonify({"error": "No image selected."}), 400

        try:
            image_bytes = BytesIO(file_storage.read())
            image = Image.open(image_bytes).convert("RGB")
        except Exception:
            return jsonify({"error": "Invalid image file."}), 400

        try:
            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

            with torch.no_grad():
                output = model.generate(**inputs, max_length=50)

            caption_text = processor.decode(output[0], skip_special_tokens=True)
            return jsonify({"caption": caption_text})
        except Exception as exc:
            return jsonify({"error": f"Failed to generate caption: {exc}"}), 500

    return app


app = create_app()


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=debug_mode)
