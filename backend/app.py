from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from temp import test
from RagPythonLocal.rag_user_chat import initialize_user_chat, input_query

app = Flask(__name__)
CORS(app)

@app.route("/api/generate", methods=["POST"])
def handle_generate():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        #Enforce rules below
        prompt = data.get("prompt", "")
        rag = data.get("rag", False)

        #Set this as bool for if historyWindow==0: false
        history = data.get("history", [])

        #Force int
        topK = data.get("topK", 5)

        #force int
        historyWindow = data.get("historyWindow", 10)

        #if filter is on, use similarity threshold. If off, set threshold to 0.0
        filter = data.get("filter", True)
        similarityThreshold = data.get("similarityThreshold", 0.4)

        #If rag is off, set bm25 to false
        bm25 = data.get("bm25", False)

        #If rag is off, set topic retreival to false
        topic_retreival = data.get()



        if not prompt:
            return jsonify({"error": "No data provided"}), 400

        response_text = test([prompt, rag, history, topK, historyWindow, filter, similarityThreshold, bm25])
        #Fill in dictionary as prompt
        response_text = input_query(initialize_user_chat(UID), {})
        # response_text = prompt_model(prompt, rag)[0].text
        
        return jsonify({"response": response_text})

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        print("Error processing request: {}".format(str(e)))
        return jsonify({"error": "Internal server error"}), 500