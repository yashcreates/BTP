from flask import Flask, render_template, request
import requests
import os
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from serpapi import GoogleSearch

# Initialize Flask app
app = Flask(__name__)

# Load AI models
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Load data
corpus_data = pd.read_csv('./summarized_corpus.csv')
with open('./adjacency_list.json', 'r') as file:
    adjacency_list = json.load(file)

# Skill extraction using Gemini API
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

# Generate detailed explanation using Gemini API
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

# SentenceTransformer embedding
def get_sbert_embedding(text):
    return sbert_model.encode(text)

# T5 semantic similarity
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

# Find relevant people using SBERT and T5
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

# Search academic papers using SerpAPI
def get_scholar_results(query):
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": os.getenv("SERPAPI_KEY"),
        "num": 5
    }
    search = GoogleSearch(params)
    return search.get_dict().get('organic_results', [])

# Search patents using SerpAPI
def get_patent_results_serpapi(query):
    params = {
        "engine": "google_patents",
        "q": query,
        "api_key": os.getenv("SERPAPI_KEY")
    }
    search = GoogleSearch(params)
    return search.get_dict().get("organic_results", [])

# Process user input
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

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    prompt = request.form.get('prompt')
    result = process_input(prompt)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)