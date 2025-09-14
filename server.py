# Bena Smith 03/28/2025
# This is a flask server that takes a form that a user uploads, parses the information,
# uses open AI's sentence embedding API, and compares the embedding to the stored 
# embeddings of previous users. Then the most similar users are returned. Open AI is also prompted
# to return potential activities that these people would all enjoy

# Sentence Embedding code from 
# https://www.datacamp.com/tutorial/introduction-to-text-embeddings-with-the-open-ai-api
# https://campus.datacamp.com/courses/introduction-to-embeddings-with-the-openai-api/embeddings-for-ai-applications?ex=5

# Bena Smith 03/28/2025
# Flask server for user matching and Spotify integration

from flask import Flask, request, jsonify, make_response, redirect, session, url_for
from flask_cors import CORS
import pandas as pd
import openai
from scipy.spatial import distance
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.cluster import KMeans
import json
from dotenv import load_dotenv
import os
import urllib.parse
import requests

load_dotenv()  # Load variables from .env

app = Flask(__name__)

# ✅ Proper CORS setup (allow only your frontend, with credentials)
CORS(
    app,
    supports_credentials=True,
    origins=["https://pair-recommender-client-6rb88.ondigitalocean.app"]
)

app.secret_key = os.getenv("FLASK_SECRET_KEY")

SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=OPENAI_API_KEY)


# ---------------------------
#   Health + test routes
# ---------------------------
@app.route("/api", methods=['GET'])
def api_test():
    return "Server is running!", 200

@app.route("/", methods=['GET'])
def home():
    return "Server is running!", 200

@app.route("/test", methods=['GET'])
def test():
    return "Test!", 200

@app.route('/health', methods=['GET'])
def health_check():
    return "OK", 200


# ---------------------------
#   Submit form + embeddings
# ---------------------------
@app.route('/api/submit_form', methods=['POST'])
def submit():
    data = request.get_json()

    our_people = pd.read_csv("our_people.csv")  # get existing profiles

    new_person_embeddings = get_new_person_embeddings(data, our_people)
    other_embeddings = our_people["Embeddings"]
    similar_embeddings = find_n_closest(new_person_embeddings, other_embeddings, 5)
    similar_people = find_similar_people(similar_embeddings, our_people)

    event_suggestions = get_event(data, similar_people)

    response = jsonify({
        'message': 'Data received successfully!',
        'similar_people': similar_people,
        'event_suggestions': event_suggestions
    })
    return response


# ---------------------------
#   Embeddings helpers
# ---------------------------
def get_new_person_embeddings(new_person, our_people):
    new_person["ID"] = len(our_people) + 1

    new_person["Outgoing"] = convert_to_text(new_person["Outgoing"] + 1, "Outgoing")
    new_person["Outdoorsy"] = convert_to_text(new_person["Outdoorsy"] + 1, "Outdoorsy")
    new_person["Politically_Correct"] = convert_to_text(new_person["Politically_Correct"] + 1, "Politically_Correct")

    about_me_embedding = np.array(create_embeddings([new_person['About_Me']])[0])
    music_embedding = np.array(create_embeddings(str([new_person['Favorite_Music_Genres']]))[0])
    outgoing_embedding = np.array(create_embeddings([new_person['Outgoing']])[0])
    outdoorsy_embedding = np.array(create_embeddings([new_person['Outdoorsy']])[0])
    politically_correct_embedding = np.array(create_embeddings([new_person['Politically_Correct']])[0])
    religion_embedding = np.array(create_embeddings([new_person['Religion']])[0])

    combined_embedding = (
        weights['About_Me'] * about_me_embedding +
        weights['Favorite_Music_Genres'] * music_embedding +
        weights['Outgoing'] * outgoing_embedding +
        weights['Outdoorsy'] * outdoorsy_embedding +
        weights['Politically_Correct'] * politically_correct_embedding +
        weights['Religion'] * religion_embedding
    )
    
    return combined_embedding


def add_to_csv(new_person, our_people):
    our_people.loc[len(our_people)] = new_person
    # our_people.to_csv('output.csv', index=False)


def convert_to_text(value, feature_name):
    if feature_name == "Outgoing":
        return ["Very introverted", "Somewhat introverted", "Neutral", "Somewhat outgoing", "Very outgoing"][int(value - 1)]
    elif feature_name == "Outdoorsy":
        return ["Hates outdoors", "Prefers indoors", "Neutral", "Likes outdoors", "Loves outdoors"][int(value - 1)]
    elif feature_name == "Politically_Correct":
        return ["Not PC at all", "Rarely PC", "Moderately PC", "Very PC", "Extremely PC"][int(value - 1)]


weights = {
    'About_Me': 1.0,
    'Favorite_Music_Genres': 0.5,
    'Outgoing': 0.3,
    'Outdoorsy': 0.3,
    'Politically_Correct': 0.3,
    'Religion': 0.5
}


def create_embeddings(about_mes):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=about_mes
    )
    response_dict = response.model_dump()
    return [data['embedding'] for data in response_dict['data']]


def find_n_closest(query_vector, embeddings, n=3):
    distances = []
    for index, embedding in enumerate(embeddings):
        embedding = embedding.strip('[]').split(',')
        embedding = np.array(embedding, dtype=np.float64)
        dist = distance.cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index})
    distances_sorted = sorted(distances, key=lambda x: x["distance"])
    return distances_sorted[0:n]
        

def find_similar_people(similar_embeddings, our_people):
    similar_people = []
    for person in similar_embeddings:
        index = person["index"]
        person_row = our_people.iloc[index]
        person_row = person_row.drop("Embeddings")
        similar_people.append(person_row.to_dict())
    return similar_people


def get_event(current_person, similar_people):
    prompt = (
        f"""Here are some profiles of people: {current_person}, {similar_people}. You run a company that 
        matches people up and comes up with an event for them to do. ...
        """
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# ---------------------------
#   Spotify OAuth
# ---------------------------
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE_URL = "https://api.spotify.com/v1"
SPOTIFY_SCOPE = "playlist-read-private playlist-read-collaborative"

@app.route("/spotify/login")
def spotify_login():
    auth_query = {
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "scope": SPOTIFY_SCOPE,
        "client_id": SPOTIFY_CLIENT_ID
    }
    url_args = urllib.parse.urlencode(auth_query)
    auth_url = f"{SPOTIFY_AUTH_URL}?{url_args}"
    return redirect(auth_url)

@app.route("/spotify/callback")
def spotify_callback():
    code = request.args.get("code")
    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "client_id": SPOTIFY_CLIENT_ID,
        "client_secret": SPOTIFY_CLIENT_SECRET,
    }
    r = requests.post(SPOTIFY_TOKEN_URL, data=token_data)
    token_info = r.json()

    session["access_token"] = token_info.get("access_token")
    session["refresh_token"] = token_info.get("refresh_token")

    return redirect("https://pair-recommender-client-6rb88.ondigitalocean.app")

@app.route("/spotify/playlists")
def spotify_playlists():
    access_token = session.get("access_token")
    if not access_token:
        return redirect(url_for("spotify_login"))
    
    headers = {"Authorization": f"Bearer {access_token}"}
    r = requests.get(f"{SPOTIFY_API_BASE_URL}/me/playlists", headers=headers)
    return jsonify(r.json())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)




### following 3 functions are throwing the kitchen sink at a cors error so I can post to submit!!!
# Niels B. on stack overflow https://stackoverflow.com/questions/25594893/how-to-enable-cors-in-flask
# @app.before_request
# def before_request():
#     headers = {'Access-Control-Allow-Origin': '*',
#                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
#                'Access-Control-Allow-Headers': 'Content-Type'}
#     if request.method.lower() == 'options':
#         return jsonify(headers)

# @app.after_request
# def add_cors_headers(response):
#     response.headers["Access-Control-Allow-Origin"] = "*"
#     response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
#     response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
#     return response

# def _build_cors_preflight_response():
#     response = make_response()
#     response.headers.add("Access-Control-Allow-Origin", "*")
#     response.headers.add('Access-Control-Allow-Headers', "*")
#     response.headers.add('Access-Control-Allow-Methods', "*")
#     return response