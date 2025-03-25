from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit, join_room
import requests
import os
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from serpapi import GoogleSearch
from dotenv import load_dotenv
from collections import deque
from pymongo import MongoClient
from bson import ObjectId
from flask_cors import CORS
from flask_mail import Mail, Message

# Load environment variables
load_dotenv()

# Initialize Flask app with SocketIO
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'your_secret_key_here'
socketio = SocketIO(app, cors_allowed_origins="*")

CORS(app)

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587  # For TLS
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'shan.tanu.rank@gmail.com'  # Your email address
app.config['MAIL_PASSWORD'] = 'qgjqqxhtcfbqnmjw'  # Your email password
app.config['MAIL_DEFAULT_SENDER'] = 'shan.tanu.rank@gmail.com'

# Initialize the Mail object
mail = Mail(app)

# Connect to MongoDB
client = MongoClient("mongodb+srv://shantanurankhambe:rkKLvTaAf4USzeL9@cluster0.wg78u.mongodb.net/")
db = client["BTP"]
collection = db["Requests"]

# Load AI models (unchanged)
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Load data (unchanged)
corpus_data = pd.read_csv('./summarized_corpus.csv')
with open('./adjacency_list.json', 'r') as file:
    adjacency_list = json.load(file)

# Track online users
online_users = {}  # sid: username

# Skill extraction using Gemini API (unchanged)
def extract_skills_from_prompt(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={os.getenv('GEMINI_API_KEY')}"
    headers = {"Content-Type": "application/json"}
    
    refined_prompt = f"I need a guy who can help me in {prompt}. Give me in a single paragraph the environmental, technical, and non-tech skills the person should have and the preferred years of experience. Only use nouns."
    
    payload = {"contents": [{"parts": [{"text": refined_prompt}]}]}
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    
    print(f"Gemini API Error: {response.status_code}")
    return ""

# Generate detailed explanation using Gemini API (unchanged)
def generate_why_and_how_explanation_gemini(person, skill, query):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={os.getenv('GEMINI_API_KEY')}"
    headers = {"Content-Type": "application/json"}
    corpus_text = person['Corpus']
    prompt = (
        f"To solve the problem: '{query}', why is this person with his company and resources best suited based on the given info, and how can they help, which part precisely I can ask them help for?\n"
        f"Person's Details (or his company's details): {corpus_text}\n"
        f"Speciality in: {skill}\n"
        f"Provide a 5-6 line explanation and it should only be positive."
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    print(f"Gemini API Error: {response.status_code}")
    return "Detailed explanation unavailable due to API error."

# SentenceTransformer embedding (unchanged)
def get_sbert_embedding(text):
    return sbert_model.encode(text)

# T5 semantic similarity (unchanged)
def t5_semantic_similarity(query, document):
    input_text = f"mnli premise: {query} hypothesis: {document}"
    inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = t5_model.generate(**inputs, max_length=3)
    score = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "entailment" in score.lower():
        return 1.0
    elif "neutral" in score.lower():
        return 0.5
    return 0.0

# Find relevant people using SBERT and T5 (unchanged)
def find_relevant_people_sbert_t5_v3(extracted_text, corpus_data, adjacency_list, query):
    skills = extracted_text.lower().split()
    skills = list(set(skills))
    skill_embeddings = sbert_model.encode(skills)
    avg_skill_embedding = skill_embeddings.mean(axis=0)

    relevant_people = []
    for idx, row in corpus_data.head(3).iterrows():
        corpus_text = row['Corpus']
        corpus_embedding = get_sbert_embedding(corpus_text)
        cosine_score = cosine_similarity([avg_skill_embedding], [corpus_embedding])[0][0]
        t5_score = t5_semantic_similarity(" ".join(skills), corpus_text)
        combined_score = (0.7 * cosine_score) + (0.3 * t5_score) + 0.3

        if combined_score > 0.01:
            person_info = {
                'Name': row['LinkedIn Name'],
                'Corpus': corpus_text,
                'Matching Skills': skills,
                'Cosine Similarity': cosine_score,
                'T5 Semantic Similarity': t5_score,
                'Combined Similarity Score': combined_score,
                'Why They Can Help': f"{row['LinkedIn Name']} has expertise in {', '.join(skills)}.",
            }
            detailed_explanation = generate_why_and_how_explanation_gemini(person_info, ", ".join(skills), query)
            person_info['Detailed Explanation'] = detailed_explanation
            relevant_people.append(person_info)

    return sorted(relevant_people, key=lambda x: x['Combined Similarity Score'], reverse=True)[:3]

# Search academic papers using SerpAPI (unchanged)
def get_scholar_results(query):
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": os.getenv("SERPAPI_KEY"),
        "num": 5
    }
    search = GoogleSearch(params)
    return search.get_dict().get('organic_results', [])

# Search patents using SerpAPI (unchanged)
def get_patent_results_serpapi(query):
    params = {
        "engine": "google_patents",
        "q": query,
        "api_key": os.getenv("SERPAPI_KEY")
    }
    search = GoogleSearch(params)
    return search.get_dict().get("organic_results", [])

# Process user input (unchanged)
def process_input(prompt):
    extracted_text = extract_skills_from_prompt(prompt)
    if not extracted_text:
        return {"error": "Failed to extract skills."}
    
    relevant_people = find_relevant_people_sbert_t5_v3(extracted_text, corpus_data, adjacency_list, prompt)
    paper_results = get_scholar_results(prompt)
    patent_results = get_patent_results_serpapi(prompt)
    
    return {
        "expert": relevant_people[0] if relevant_people else None,
        "paper_results": paper_results,
        "patent_results": patent_results
    }

# Flask routes (unchanged)
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    prompt = request.form.get('prompt')
    result = process_input(prompt)
    return render_template('result.html', result=result)

# SocketIO events for chat
@socketio.on('join')
def handle_join(data):
    username = data['username']
    online_users[request.sid] = username
    join_room('public')
    emit('online_users', list(online_users.values()), broadcast=True)
    print(f"User {username} joined with SID {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    username = online_users.pop(request.sid, None)
    if username:
        emit('online_users', list(online_users.values()), broadcast=True)
        print(f"User {username} disconnected")

@socketio.on('request_private_chat')
def handle_request_private_chat(data):
    target_username = data['target']
    requester_username = online_users.get(request.sid)
    if requester_username and target_username in online_users.values():
        for sid, user in online_users.items():
            if user == target_username:
                emit('private_chat_request', {'from': requester_username, 'to': target_username}, room=sid)
                print(f"Chat request from {requester_username} to {target_username} sent to SID {sid}")
                break
    else:
        print(f"Failed to send request: {requester_username} or {target_username} not found")

@socketio.on('accept_private_chat')
def handle_accept_private_chat(data):
    requester_username = data['requester']
    acceptor_username = online_users.get(request.sid)
    if acceptor_username and requester_username in online_users.values():
        room_name = '_'.join(sorted([requester_username, acceptor_username]))
        join_room(room_name, sid=request.sid)
        print(f"{acceptor_username} joined room {room_name} with SID {request.sid}")
        for sid, user in online_users.items():
            if user == requester_username:
                join_room(room_name, sid=sid)
                emit('private_chat_started', {'room': room_name, 'partner': acceptor_username}, room=sid)
                print(f"{requester_username} joined room {room_name} with SID {sid}")
                emit('private_chat_started', {'room': room_name, 'partner': requester_username}, room=request.sid)
                break
    else:
        print(f"Failed to start private chat: {requester_username} or {acceptor_username} not found")

@socketio.on('send_message')
def handle_send_message(data):
    room = data['room']
    message = data['message']
    username = online_users.get(request.sid)
    if username and message:
        formatted_message = f"{username}: {message}"
        emit('chat_message', {'room': room, 'message': formatted_message}, room=room)
        print(f"Message '{formatted_message}' sent to room {room}")
    else:
        print(f"Failed to send message: User {username} or message {message} invalid")

#===============================================================================================================

# SSR

# Load the adjacency list
def load_adjacency_list(file_path):
    try:
        with open(file_path, 'r') as f:
            adjacency_list = json.load(f)  # Ensure your file is JSON-formatted
        return adjacency_list
    except Exception as e:
        print(f"Error loading adjacency list: {e}")
        return {}

ADJACENCY_LIST_FILE = './adjacency_list_with_profession.json'
adjacency_list = load_adjacency_list(ADJACENCY_LIST_FILE)
file_path = './data.xlsx'
profession_data = pd.read_excel(file_path)
name_to_profession = dict(zip(profession_data['LinkedIn Name'], profession_data['Description']))

# Perform BFS search for person
def bfs_with_person(start, target):
    queue = deque([(start, [start])])
    visited = set()
    visited.add(start)

    while queue:
        current, path = queue.popleft()

        # Check if the person matches
        if current.strip().lower() == target.strip().lower():
            return path

        for neighbor, _ in adjacency_list.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None

# Perform BFS search for profession
def bfs_with_profession(start, profession):
    queue = deque([(start, [start])])
    visited = set()
    visited.add(start)

    while queue:
        current, path = queue.popleft()

        if name_to_profession.get(current, '') == profession:
            return path

        for neighbor, _ in adjacency_list.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None

@app.route('/searchprofession/<string:start>/<string:target>', methods=['GET'])
def searchprofession(start, target):

    if not start:
        return jsonify({"error": "Start name is required"}), 400

    if target:
        result = bfs_with_profession(start, target)
        if result:
            return jsonify({"path": " -> ".join(result)})
        else:
            return jsonify({"error": "No connection found"}), 404

    return jsonify({"error": "No valid target or profession provided"}), 400

@app.route('/searchperson/<string:start>/<string:target>', methods=['GET'])
def searchperson(start, target):

    if not start:
        return jsonify({"error": "Start name is required"}), 400

    if target:
        result = bfs_with_person(start, target)
        if result:
            return jsonify({"path": " -> ".join(result)})
        else:
            return jsonify({"error": "No connection found"}), 404

    return jsonify({"error": "No valid target or profession provided"}), 400

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    start = data.get('start')
    target = data.get('target')
    profession = data.get('profession')

    if not start:
        return jsonify({"error": "Start name is required"}), 400

    # Search by person
    if target:
        result = bfs_with_person(start, target)
        if result:
            return jsonify({"path": " -> ".join(result)})
        else:
            return jsonify({"error": "No connection found"}), 404

    # Search by profession
    if profession:
        result = bfs_with_profession(start, profession)
        if result:
            return jsonify({"path": " -> ".join(result)})
        else:
            return jsonify({"error": "No connection found for that profession"}), 404

    return jsonify({"error": "No valid target or profession provided"}), 400

# Serve the adjacency list
@app.route('/adjacency-list', methods=['GET'])
def get_adjacency_list():
    return jsonify(adjacency_list)

@app.route('/add')
def add_data():
    collection.insert_one({"name": "John", "age": 30})
    return jsonify(message="User added!")

@app.route('/add_request', methods=['POST'])
def add_request():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid data"}), 400
    
    collection.insert_one({
        "from": data["from"],
        "to": data["to"],
        "project_title": data["project_title"],
        "project_description": data["project_description"],
        "status": "pending"
    })
    
    return jsonify({"success": True, "message": "Request added successfully!"})

@app.route('/get_responses/<userName>', methods=['GET'])
def get_responses(userName):
    requests = list(collection.find({"from": userName, "status": "accepted"}, {"_id": 1, "from": 1, "to": 1, "project_title": 1, "project_description": 1, "status": 1, "hours_per_week": 1, "response_message": 1}))

    # Convert `_id` to string so JavaScript can use it
    for req in requests:
        req["_id"] = str(req["_id"])  # Convert ObjectId to string

    print("Returning Requests:", requests)  # Debugging: Check if `_id` is sent correctly
    return jsonify(requests)

@app.route('/get_requests/<userName>', methods=['GET'])
def get_requests(userName):
    requests = list(collection.find({"to": userName, "status": "pending"}, {"_id": 1, "from": 1, "to": 1, "project_title": 1, "project_description": 1, "status": 1}))

    # Convert `_id` to string so JavaScript can use it
    for req in requests:
        req["_id"] = str(req["_id"])  # Convert ObjectId to string

    print("Returning Requests:", requests)  # Debugging: Check if `_id` is sent correctly
    return jsonify(requests)


@app.route('/update_request', methods=['POST'])
def update_request():
    try:
        data = request.get_json(force=True)  # Force JSON parsing

        print("Received Data:", data)  # Debugging output in Flask console

        request_id = data.get("request_id")
        status = data.get("status")

        if not request_id or not status:
            return jsonify({"error": "Missing request_id or status"}), 400

        try:
            obj_id = ObjectId(request_id)
        except Exception as e:
            return jsonify({"error": "Invalid ObjectId format"}), 400

        update_data = {"status": status}

        if status == "accepted":
            userdata = collection.find_one({"_id": obj_id})
            print("userdata", userdata)

            update_data["hours_per_week"] = int(data.get("hours_per_week", 0))  # Convert to int
            update_data["response_message"] = data.get("response_message", "")

            recipient_email = userdata["from"]
            subject = "Update regarding your connection request"
            body = f"""
                Your request to {userdata["to"]} got approved.
                Availability: {int(data.get("hours_per_week", 0))} hours.
                Message:
                {data.get("response_message", "")}
            """
            msg = Message(subject=subject,
                        recipients=[recipient_email],
                        body=body)
            try:
                mail.send(msg)
                # return 'Email sent successfully!'
            except Exception as e:
                print("Email not sent")
                # return f"Failed to send email. Error: {e}"

        result = collection.update_one({"_id": obj_id}, {"$set": update_data})

        if result.matched_count == 0:
            return jsonify({"error": "Request not found"}), 404

        return jsonify({"success": True, "message": f"Request {status} successfully!"})
    
    except Exception as e:
        print("Error:", str(e))  # Debugging output
        return jsonify({"error": str(e)}), 500  # Catch unexpected errors



#===============================================================================================================

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5080, debug=True)
