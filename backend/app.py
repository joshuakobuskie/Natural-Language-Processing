from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from temp import test
from rag_user_chat import initialize_user_chat, input_query, end_user_session

app = Flask(__name__)
CORS(app)

def user_rag_query(user_output_folder, saved_chats_topic_name, frontend_inputs, task_folder='saved_chats'):
    user_chat_settings = initialize_user_chat(task_folder, user_output_folder, saved_chats_topic_name)
    user_chat_settings = input_query(user_chat_settings, frontend_inputs)
    end_user_session(user_chat_settings)
    return user_chat_settings['user_query_state_history'][max(user_chat_settings['user_query_state_history'])]['response_text']

@app.route("/api/generate", methods=["POST"])
def handle_generate():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        
        prompt = data.get("prompt", "")
        rag = data.get("rag", False)
        topK = data.get("topK", 5)
        historyWindow = data.get("historyWindow", 10)
        filter = data.get("filter", True)
        similarityThreshold = data.get("similarityThreshold", 0.4)
        bm25 = data.get("bm25", False)
        topicRetreival = data.get("topicRetrieval", False)

        #Enforcing types and rules
        prompt = str(prompt)
        rag = bool(rag)
        topK = int(topK)
        historyWindow = int(historyWindow)
        filter = bool(filter)
        similarityThreshold = float(similarityThreshold)
        bm25 = bool(bm25)
        topicRetreival = bool(topicRetreival)

        if not rag:
            bm25 = False
            topicRetreival = False

        if not filter:
            similarityThreshold = 0.0
        
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