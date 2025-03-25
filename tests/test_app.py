import pytest
import json
from app import app, db, collection, get_sbert_embedding, process_input, extract_skills_from_prompt
from flask_socketio import SocketIOTestClient
from pymongo import MongoClient
from flask import Flask
from unittest.mock import patch
from bson import ObjectId

@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        yield client

@pytest.fixture
def socket_client():
    client = SocketIOTestClient(app, flask_test_client=app.test_client())
    yield client
    client.disconnect()

@pytest.fixture
def mock_db():
    with patch("app.collection") as mock_collection:
        yield mock_collection

# ========================== #
# 1. Basic Flask Route Tests #
# ========================== #

def test_home(client):
    response = client.get("/")
    assert response.status_code == 200

def test_index(client):
    response = client.get("/index")
    assert response.status_code == 200

def test_analyze(client):
    response = client.post("/analyze", data={"prompt": "AI and NLP"})
    assert response.status_code == 200

# ============================ #
# 2. MongoDB Request Handling #
# ============================ #

def test_add_request(client, mock_db):
    mock_db.insert_one.return_value = None
    data = {
        "from": "UserA",
        "to": "UserB",
        "project_title": "AI Project",
        "project_description": "Description"
    }
    response = client.post("/add_request", json=data)
    assert response.status_code == 200
    assert response.json["success"] is True

def test_get_requests(client, mock_db):
    mock_db.find.return_value = [{"_id": ObjectId(), "from": "UserA", "project_title": "AI Project"}]
    response = client.get("/get_requests/UserB")
    assert response.status_code == 200
    assert len(response.json) > 0

def test_update_request(client, mock_db):
    mock_db.update_one.return_value.matched_count = 1
    data = {
        "request_id": str(ObjectId()),
        "status": "accepted",
        "hours_per_week": 5,
        "response_message": "Let's collaborate!"
    }
    response = client.post("/update_request", json=data)
    assert response.status_code == 200
    assert response.json["success"] is True

# ===================================== #
# 3. AI Model and Skill Extraction Tests #
# ===================================== #

def test_get_sbert_embedding():
    embedding = get_sbert_embedding("Artificial Intelligence")
    assert len(embedding) > 0

@patch("app.requests.post")
def test_extract_skills_from_prompt(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Machine Learning, NLP"}]}}]
    }
    skills = extract_skills_from_prompt("I need an AI expert")
    assert "Machine Learning" in skills

# ========================== #
# 4. Socket.IO Chat Tests #
# ========================== #

def test_socketio_join(socket_client):
    socket_client.emit("join", {"username": "User1"})
    received = socket_client.get_received()
    assert any(event["name"] == "online_users" for event in received)

def test_socketio_private_chat(socket_client):
    socket_client.emit("join", {"username": "User1"})
    socket_client.emit("request_private_chat", {"target": "User2"})
    received = socket_client.get_received()
    assert any(event["name"] == "private_chat_request" for event in received)

def test_socketio_send_message(socket_client):
    socket_client.emit("join", {"username": "User1"})
    socket_client.emit("send_message", {"room": "test_room", "message": "Hello"})
    received = socket_client.get_received()
    assert any(event["name"] == "chat_message" for event in received)

# ================================= #
# 5. Search Person & Profession Tests #
# ================================= #

def test_search_person(client):
    response = client.get("/searchperson/Alice/Bob")
    assert response.status_code in [200, 404]

def test_search_profession(client):
    response = client.get("/searchprofession/Alice/Data Scientist")
    assert response.status_code in [200, 404]

# ============================= #
# 6. Scholar & Patent Search Tests #
# ============================= #

@patch("app.GoogleSearch")
def test_get_scholar_results(mock_search):
    mock_search.return_value.get_dict.return_value = {"organic_results": [{"title": "AI Paper"}]}
    response = process_input("AI research")
    assert "paper_results" in response

@patch("app.GoogleSearch")
def test_get_patent_results(mock_search):
    mock_search.return_value.get_dict.return_value = {"organic_results": [{"title": "AI Patent"}]}
    response = process_input("AI research")
    assert "patent_results" in response

