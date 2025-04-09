from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from temp import test
from continuous_response_generation import prompt_model

app = Flask(__name__)
CORS(app)

@app.route("/api/generate", methods=["POST"])
def handle_generate():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        prompt = data.get("prompt", "")
        rag = data.get("rag", False)
        history = data.get("history", [])

        if not prompt:
            return jsonify({"error": "No data provided"}), 400

        # response_text = test(prompt, rag, history)
        response_text = prompt_model(prompt, rag)[0].text
        
        return jsonify({"response": response_text})

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        print("Error processing request: {}".format(str(e)))
        return jsonify({"error": "Internal server error"}), 500