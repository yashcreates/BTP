from flask import Flask, render_template, request
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

# Load environment variables
load_dotenv()

# Initialize Flask app with SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Replace with a secure key
socketio = SocketIO(app, cors_allowed_origins="*")

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
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
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
        f"Provide a 5-6 line explanation."
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
        combined_score = (0.6 * cosine_score) + (0.4 * t5_score) + 0.3

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

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5080, debug=True)
