import json
import re
import random
import pickle
import requests
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

# Load necessary files
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Movie details dictionary
movie_details = {
    "mad max: fury road": "'Mad Max: Fury Road' is a post-apocalyptic action film directed by George Miller. Set in a desert wasteland where water and gasoline are scarce, it follows Max Rockatansky who is captured by Immortan Joe's War Boys. Along with rebel warrior Furiosa, Max tries to escape Joe's tyrannical rule. The film is known for its high-octane action sequences, stunning visuals, and intense performances.",
    "john wick": "'John Wick' is an action-thriller film starring Keanu Reeves. It follows John Wick, a retired hitman seeking vengeance for the murder of his dog, a final gift from his deceased wife. The movie delves deep into the criminal underworld and is renowned for its stylized action sequences, choreographed fights, and the stoic performance of Reeves.",
    "die hard": "'Die Hard' is an action film starring Bruce Willis as NYPD officer John McClane. Set on Christmas Eve, McClane fights to save hostages, including his wife, from terrorists who have taken over the Nakatomi Plaza in Los Angeles. The film is celebrated for its suspenseful action, memorable one-liners, and Willis' portrayal of an everyman hero.",
    "the dark knight": "'The Dark Knight' is a superhero film directed by Christopher Nolan, featuring Batman as he battles the Joker, a criminal mastermind who seeks to plunge Gotham City into chaos. The film is known for its dark tone, complex characters, and Heath Ledger's iconic performance as the Joker, which earned him a posthumous Academy Award.",
    "superbad": "'Superbad' is a comedy film about high school friends Seth and Evan, who go on a wild adventure to procure alcohol for a party in hopes of impressing their crushes before graduation. The film is a coming-of-age story filled with humor, awkward moments, and a heartfelt exploration of friendship.",
    "the hangover": "'The Hangover' is a comedy film about three friends who retrace their steps after a wild bachelor party in Las Vegas to find their missing friend before his wedding. Known for its outrageous situations, memorable characters, and comedic chaos, the film was a huge success and spawned several sequels.",
    "step brothers": "'Step Brothers' is a comedy film about two middle-aged men, Brennan and Dale, who become stepbrothers when their single parents marry. The film explores their rivalry, antics, and eventual friendship. It is filled with humor, absurd situations, and the dynamic comedic performances of Will Ferrell and John C. Reilly.",
    "anchorman": "'Anchorman: The Legend of Ron Burgundy' is a comedy film about Ron Burgundy, a top-rated news anchor in 1970s San Diego. The story follows Burgundy and his news team as they face challenges posed by the arrival of a talented female reporter. The film is known for its satirical take on the news industry and its quirky characters.",
    "the notebook": "'The Notebook' is a romantic drama about a young couple, Noah and Allie, who fall in love during the early 1940s. The film chronicles their passionate romance, societal pressures, and enduring love despite various obstacles. It is celebrated for its emotional depth, beautiful storytelling, and the chemistry between its lead actors.",
    "pride and prejudice": "'Pride and Prejudice' is a romantic drama based on Jane Austen's novel. It follows the life of Elizabeth Bennet as she navigates issues of manners, morality, and marriage in early 19th century England. The film is known for its sharp wit, beautiful period costumes, and the timeless romance between Elizabeth and Mr. Darcy.",
    "la la land": "'La La Land' is a romantic musical film about an aspiring actress, Mia, and a jazz musician, Sebastian, as they pursue their dreams in Los Angeles. Directed by Damien Chazelle, the film is noted for its vibrant cinematography, memorable music, and a bittersweet love story that explores the sacrifices made for artistic ambitions.",
    "titanic": "'Titanic' is a romantic disaster film directed by James Cameron. It tells the fictionalized story of the RMS Titanic's ill-fated voyage and the romance between Jack, a poor artist, and Rose, a wealthy young woman. The film is known for its epic scale, visual effects, and the tragic love story that unfolds against the backdrop of the ship's sinking.",
    "inception": "'Inception' is a sci-fi thriller directed by Christopher Nolan. The film follows Dom Cobb, a thief who enters the dreams of others to steal secrets. Cobb is given a chance to have his criminal history erased if he can successfully plant an idea into someone's subconscious. Known for its complex narrative structure and stunning visual effects, 'Inception' explores themes of reality and dreams.",
    "interstellar": "'Interstellar' is a sci-fi film directed by Christopher Nolan. It follows a group of astronauts who travel through a wormhole near Saturn in search of a new habitable planet for humanity as Earth faces environmental collapse. The film is acclaimed for its scientific accuracy, emotional depth, and impressive visual effects.",
    "the matrix": "'The Matrix' is a sci-fi action film directed by the Wachowskis. It follows Neo, a computer hacker who discovers that reality as he knows it is a simulation created by sentient machines to subdue humanity. The film is renowned for its groundbreaking special effects, philosophical themes, and innovative action sequences.",
    "blade runner 2049": "'Blade Runner 2049' is a sci-fi film directed by Denis Villeneuve. It is a sequel to the 1982 film 'Blade Runner'. The story follows K, a replicant 'blade runner' who uncovers a secret that threatens to destabilize society. The film is noted for its stunning visual style, atmospheric world-building, and thought-provoking themes about humanity and artificial intelligence.",
    "the conjuring": "'The Conjuring' is a horror film directed by James Wan. It is based on the true story of paranormal investigators Ed and Lorraine Warren, who help a family terrorized by a dark presence in their farmhouse. The film is praised for its suspenseful atmosphere, effective scares, and strong performances.",
    "get out": "'Get Out' is a horror film written and directed by Jordan Peele. The film follows Chris, a young African American man who uncovers shocking secrets when he meets the family of his white girlfriend. 'Get Out' is acclaimed for its social commentary on race relations, its blend of horror and satire, and Peele's direction.",
    "hereditary": "'Hereditary' is a psychological horror film directed by Ari Aster. The story revolves around a family that begins to unravel following the death of their secretive grandmother. The film is known for its unsettling atmosphere, disturbing imagery, and powerful performances, particularly by Toni Collette.",
    "a quiet place": "'A Quiet Place' is a horror film directed by John Krasinski. Set in a post-apocalyptic world, it follows a family forced to live in silence while hiding from creatures that hunt by sound. The film is praised for its innovative use of sound, tension-filled narrative, and strong emotional core."
}

import requests

import requests

import requests

def get_imdb_rating(movie_name):
    api_key = "bba2887f"  # Your OMDb API key
    url = f"http://www.omdbapi.com/?t={movie_name}&apikey={api_key}"
    try:
        print(f"Requesting URL: {url}")  # Debugging: Print the URL
        response = requests.get(url, timeout=5)  # Set a timeout for the request
        print(f"Response status code: {response.status_code}")  # Debugging: Print response status code
        data = response.json()
        print(f"Response data: {data}")  # Debugging: Print the response data
        if data['Response'] == 'True':
            short_url = f"/imdb/{data['imdbID']}"
            return data['imdbRating'], short_url
        else:
            return "not found", None
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}", None

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def extract_movie_name(sentence):
    for movie in movie_details.keys():
        if all(word in sentence.lower() for word in movie.split()):
            return movie
    return ""

def analyze_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    print(f"Sentiment scores: {sentiment}")  # Debugging: Print sentiment scores
    if sentiment['compound'] > 0.6:
        return "positive"
    elif sentiment['compound'] < -0.6:
        return "negative"
    else:
        return "neutral"


def get_response(intents_list, intents_json, user_query):
    global last_recommended_movie
    if not intents_list:
        return "I'm sorry, I don't understand."

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    sentiment = analyze_sentiment(user_query)
    
    print(f"Detected tag: {tag}")  # Debugging: Print detected tag
    print(f"Intents list: {intents_list}")  # Debugging: Print intents list
    print(f"Intents JSON: {intents_json}")  # Debugging: Print intents JSON

    for i in list_of_intents:
        if i.get('tag') == tag:
            if sentiment == "positive":
                result = random.choice(i.get('positive_responses', i['responses']))
            elif sentiment == "negative":
                result = random.choice(i.get('negative_responses', i['responses']))
            else:
                result = random.choice(i.get('neutral_responses', i['responses']))
            break
    else:
        result = "I'm sorry, I don't understand."
    
    # Movie details and IMDb rating handling
    if tag == "movie_details":
        movie_name = extract_movie_name(user_query)
        if movie_name:
            result = movie_details.get(movie_name, "I'm sorry, I don't have details about that movie.")
        elif last_recommended_movie:
            result = movie_details.get(last_recommended_movie, "I'm sorry, I don't have details about that movie.")
        else:
            result = "I'm sorry, I don't have details about that movie."
    elif tag == "imdb_rating":
        movie_name = extract_movie_name(user_query)
        if movie_name:
            rating, short_url = get_imdb_rating(movie_name)
            if short_url:
                result = f"The IMDb rating for {movie_name} is {rating}. For more details, visit: {short_url}"
            else:
                result = f"The IMDb rating for {movie_name} is {rating}."
        else:
            result = "I'm sorry, I couldn't find the IMDb rating for that movie."
    elif tag == "recommend_movie":
        result = random.choice([
            "Of course! Do you have a preferred genre or theme?",
            "Sure! What genre are you interested in?",
            "Absolutely! What kind of movie are you in the mood for?",
            "What type of movies do you like? Action, comedy, romance?",
            "Sure! Tell me what kind of movie you're in the mood for."
        ])
    elif tag.startswith("recommend_"):
        genre = tag.split("_")[1]
        genre_tag = f"recommend_{genre}"
        for i in list_of_intents:
            if i['tag'] == genre_tag:
                result = random.choice(i['responses'])
                last_recommended_movie = extract_movie_name(result)
                break

    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    if not ints:
        return "I'm not sure how to respond to that."
    res = get_response(ints, intents, msg)
    return res





def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents, msg)
    return res


# Test sentiment analysis
test_sentences = [
    "I am so happy today!",
    "This is the worst movie I have ever seen.",
    "I feel okay about this.",
]

for sentence in test_sentences:
    sentiment = analyze_sentiment(sentence)
    print(f"Input: {sentence}")
    print(f"Detected sentiment: {sentiment}")
    print("-" * 50)

# Running the chatbot
print("GO! Bot is running!")
while True:
    message = input("")
    if message.lower() == "quit":
        break
    print(chatbot_response(message))
