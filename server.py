# Bena Smith 03/28/2025
# This is a flask server that takes a form that a user uploads, parses the information,
# uses open AI's sentence embedding API, and compares the embedding to the stored 
# embeddings of previous users. Then the most similar users are returned. Open AI is also prompted
# to return potential activities that these people would all enjoy

# Sentence Embedding code from 
# https://www.datacamp.com/tutorial/introduction-to-text-embeddings-with-the-open-ai-api
# https://campus.datacamp.com/courses/introduction-to-embeddings-with-the-openai-api/embeddings-for-ai-applications?ex=5


from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
import pandas as pd
import openai
from scipy.spatial import distance
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.cluster import KMeans
import json
from flask import Flask
from dotenv import load_dotenv
import os


load_dotenv()  # Load variables from .env

app = Flask(__name__)
CORS(app, origins= "https://pair-recommender-client-6rb88.ondigitalocean.app")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=OPENAI_API_KEY)

@app.route("/")
def home():
    return "Server is running!", 200

@app.route('/health', methods=['GET'])
def health_check():
    return "OK", 200

#When a user submits a form, return similar users and some potential activities
@app.route('/submit', methods=['POST', 'GET'])
@cross_origin
def submit():
    if request.method == 'POST':

        data = request.get_json()

        our_people = pd.read_csv("../notebooks/our_people.csv") # get existing profiles

        #print("Current person: ", data, "\n")

        new_person_embeddings = get_new_person_embeddings(data, our_people) # our_people is just sent in to get a unique id for the new person
        other_embeddings = our_people["Embeddings"]
        similar_embeddings = find_n_closest(new_person_embeddings, other_embeddings, 5) # get the top 5 closest embeddings to the current person
        similar_people = find_similar_people(similar_embeddings, our_people) # get the associated people
        #members(similar_people)

        event_suggestions = get_event(data, similar_people) # get suggested events

        #add_to_csv(data, our_people) add the new person to our csv. in real life we would add the user to a database

        return jsonify({'message': 'Data received successfully!', 'similar_people': similar_people, 'event_suggestions' : event_suggestions}), 200

    if request.method == 'GET':
        print("hello")

# This function gets a weighted embedding of for a persons about me and personal traits. 
# new_person is a dictionary with the information about a person
# our_people is the table of other users. This is just used to get the person's id
def get_new_person_embeddings(new_person, our_people):
    
    new_person["ID"] = len(our_people) + 1

    # change the number values by +1 because the client collects them from 0-4 but I think 1-5 
    # makes the variable more interpretable IMO so that's how they are stored 
    # Then convert the numbers to text so we can use them to get text embeddings
    new_person["Outgoing"] = convert_to_text(new_person["Outgoing"] + 1, "Outgoing")
    new_person["Outdoorsy"] = convert_to_text(new_person["Outdoorsy"] + 1, "Outdoorsy")
    new_person["Politically_Correct"] = convert_to_text(new_person["Politically_Correct"] + 1, "Politically_Correct") 

    # get embeddings for each characteristic
    about_me_embedding = np.array(create_embeddings([new_person['About_Me']])[0])
    music_embedding = np.array(create_embeddings(str([new_person['Favorite_Music_Genres']]))[0]) # this may be a list if the person selects multiple genres. Open AI probably handles this ok but more research necessary here
    outgoing_embedding = np.array(create_embeddings([new_person['Outgoing']])[0])
    outdoorsy_embedding = np.array(create_embeddings([new_person['Outdoorsy']])[0])
    politically_correct_embedding = np.array(create_embeddings([new_person['Politically_Correct']])[0])
    religion_embedding = np.array(create_embeddings([new_person['Religion']])[0])

    # Combine and weight the embeddings
    # A stronger weight means that the characteristic is more important that two friends share.
    # If it is very important that two friends share the same politically_correct level, politically_correct 
    # should be weighted highly
    combined_embedding = (
        weights['About_Me'] * about_me_embedding +
        weights['Favorite_Music_Genres'] * music_embedding +
        weights['Outgoing'] * outgoing_embedding +
        weights['Outdoorsy'] * outdoorsy_embedding +
        weights['Politically_Correct'] * politically_correct_embedding +
        weights['Religion'] * religion_embedding
    )
    
    return combined_embedding


# this function isn't used
# it can be written out more to add the new person to the csv of our people
# this is in place of adding a new person to a database
def add_to_csv(new_person, our_people):
    our_people.loc[len(our_people)] = new_person
    #our_people.to_csv('output.csv', index=False)


# Outgoing, Outdoorsy, and Politically_Correct are stored as numbers in case I want to change the algorithm 
# or something, i think storing htese as numbers is best practice but anyhow, this function changes them to text
# representations so they can be used to get people embeddings
def convert_to_text(value, feature_name):
    if feature_name == "Outgoing":
        return ["Very introverted", "Somewhat introverted", "Neutral", "Somewhat outgoing", "Very outgoing"][int(value - 1)]
    elif feature_name == "Outdoorsy":
        return ["Hates outdoors", "Prefers indoors", "Neutral", "Likes outdoors", "Loves outdoors"][int(value - 1)]
    elif feature_name == "Politically_Correct":
        return ["Not PC at all", "Rarely PC", "Moderately PC", "Very PC", "Extremely PC"][int(value - 1)]

# These weights can be changed to represent how important these things are to someone. In practice, a user could change these
# based on what they think is important to share with another person. 
weights = {
    'About_Me': 1.0,
    'Favorite_Music_Genres': 0.5,
    'Outgoing': 0.3,
    'Outdoorsy': 0.3,
    'Politically_Correct': 0.3,
    'Religion': 0.5
}

# call open ai create embeddings 
# about_mes is a sentence/ word text about a person
def create_embeddings(about_mes):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input = about_mes
    )
    response_dict = response.model_dump()
    return [data['embedding'] for data in response_dict['data']]

# Gets the n closest embeddings 
# query_vector
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
    print("Other People:")
    for person in similar_embeddings:
        index = person["index"]
        person_row = our_people.iloc[index]
        person_row = person_row.drop("Embeddings")
        similar_people.append(person_row.to_dict())
    return(similar_people)

# current_person, similar_people are profiles of users. similar_people has multiple users. 
# Ask OpenAI to suggest some activites that these people would enjoy doing together
def get_event(current_person, similar_people):
    prompt = (
        f"""Here are some profiles of people: {current_person}, {similar_people}. You run a company that 
        matches people up and comes up with an event for them to do. You already decided that these people 
        would like eachother. Now, come up with an event that they will all like. It should be suitable 
        for the first time these people are meeting and conductive making friends. If everyone liked to go
        out and electronic music you might something like,
        \"You might like to attend an underground rave together. We can get dinner together first at a
        wine bar. \" If everyone liked playing board games you might suggest a speed puzzling night. It 
        is best to suggest something that a few people say they like to do. It is ok if not though.
        Return 3 options in a list like this "/1. You might like to attend a pottery workshop because you all like art. <br/><br/>
        2. We can meet for brunch and then go on a walk through central park because this group is contemplative and some of you like birdwatching. <br/><br/>
        3. Thrifting and coffee sounds like fun with this group because you are artsy and social."/ Feel free to be creative and come up with some
        cool activities! Also make sure to keep like <br/> html line break symbols. 
        """
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    print(response)
    return response.choices[0].message.content.strip()



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)