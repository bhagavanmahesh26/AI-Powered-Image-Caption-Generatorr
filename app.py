import os
from functools import lru_cache
from io import BytesIO
from threading import Thread

from flask import Flask, jsonify, render_template, request
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

    @lru_cache(maxsize=1)
    def load_model():
        model_name = os.getenv("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-base")
        processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        model = BlipForConditionalGeneration.from_pretrained(model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        return processor, model, device

    def _warmup_model() -> None:
        try:
            load_model()
        except Exception as exc:
            app.logger.exception("Model warmup failed: %s", exc)

    Thread(target=_warmup_model, daemon=True).start()

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
            processor, model, device = load_model()
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
