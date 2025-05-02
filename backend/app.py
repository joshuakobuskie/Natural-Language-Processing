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
        userId = data.get("userId", "temp")
        chatId = data.get("chatId", "temp")

        #Enforcing types and rules
        prompt = str(prompt)
        rag = bool(rag)
        topK = int(topK)
        historyWindow = int(historyWindow)
        filter = bool(filter)
        similarityThreshold = float(similarityThreshold)
        bm25 = bool(bm25)
        topicRetreival = bool(topicRetreival)
        userId = str(userId)
        chatId = str(chatId)

        if not rag:
            bm25 = False
            topicRetreival = False

        if not filter:
            similarityThreshold = 0.0

        if topK < 0:
            topK = 0
        
        if topK > 10:
            topK = 10

        if historyWindow < 0:
            historyWindow = 0

        if historyWindow > 100:
            historyWindow = 100

        if similarityThreshold < 0.0:
            similarityThreshold = 0.0

        if similarityThreshold > 1.0:
            similarityThreshold = 1.0
        
        if not prompt:
            return jsonify({"error": "No data provided"}), 400

        job = {
            'USER_INPUT_QUERY' : prompt,
            'DESIRED_HISTORY_WINDOW_SIZE' : historyWindow,
            'DESIRED_CONTEXT_CHUNKS_TOP_K' : topK,
            'RAG_SWITCH' : rag,
            'HISTORY_SWITCH' : filter,
            'BM25_SWITCH' : bm25,
            'TOPIC_RETRIEVAL_SWITCH' : topicRetreival,
            'HISTORIC_QUERY_SIMILARITY_THRESHOLD' : similarityThreshold,
        }
        # response_text = test([prompt, rag, topK, historyWindow, filter, similarityThreshold, bm25, topicRetreival, userId, chatId])
        
        responseText = user_rag_query(userId, chatId, job)
        
        return jsonify({"response": responseText})

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        print("Error processing request: {}".format(str(e)))
        return jsonify({"error": "Internal server error"}), 500