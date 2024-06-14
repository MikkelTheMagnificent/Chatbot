import json
import re
import random
import pickle
import requests
import numpy as np
import nltk
import spacy
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, request, jsonify, render_template
from textblob import TextBlob


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")
app = Flask(__name__)

# Load necessary files
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


# Global variables
conversation_history = []
last_recommended_movie = "" 
sentiment_scores_history = [] 
mentioned_movies = set()  


# List of movie names
movie_names = [
    "mad max: fury road", "john wick", "die hard", "the dark knight", "gladiator", # Action movies
    "superbad", "the hangover", "step brothers", "anchorman", "bridesmaids" # Comedy movies
    "the notebook", "pride and prejudice", "la la land", "titanic", "before sunrise", # Romance movies
    "inception", "interstellar", "the matrix", "blade runner 2049", "guardians of the galaxy", # Sci-fi movies
    "the conjuring", "get out", "hereditary", "a quiet place", # Horror movies
    "the shawshank redemption", "forrest gump", "the godfather", "fight club", "a beautiful mind", # Drama movies
    "toy story", "spirited away", "finding nemo", "the lion king", "up" # Animation movies

]

# Movie details dictionary 
movie_details = {
    "mad max: fury road": "'Mad Max: Fury Road' is a post-apocalyptic action film directed by George Miller. Set in a desert wasteland where water and gasoline are scarce, it follows Max Rockatansky who is captured by Immortan Joe's War Boys. Along with rebel warrior Furiosa, Max tries to escape Joe's tyrannical rule. The film is known for its high-octane action sequences, stunning visuals, and intense performances.",
    "john wick": "'John Wick' is an action-thriller film starring Keanu Reeves. It follows John Wick, a retired hitman seeking vengeance for the murder of his dog, a final gift from his deceased wife. The movie delves deep into the criminal underworld and is renowned for its stylized action sequences, choreographed fights, and the stoic performance of Reeves.",
    "die hard": "'Die Hard' is an action film starring Bruce Willis as NYPD officer John McClane. Set on Christmas Eve, McClane fights to save hostages, including his wife, from terrorists who have taken over the Nakatomi Plaza in Los Angeles. The film is celebrated for its suspenseful action, memorable one-liners, and Willis' portrayal of an everyman hero.",
    "the dark knight": "'The Dark Knight' is a superhero film directed by Christopher Nolan, featuring Batman as he battles the Joker, a criminal mastermind who seeks to plunge Gotham City into chaos. The film is known for its dark tone, complex characters, and Heath Ledger's iconic performance as the Joker, which earned him a posthumous Academy Award.",
    "gladiator": "'Gladiator' is a historical epic directed by Ridley Scott. It stars Russell Crowe as Maximus, a betrayed Roman general who seeks revenge against the corrupt emperor who murdered his family. The film is celebrated for its grand battle sequences, powerful performances, and the portrayal of ancient Rome.",
    "superbad": "'Superbad' is a comedy film about high school friends Seth and Evan, who go on a wild adventure to procure alcohol for a party in hopes of impressing their crushes before graduation. The film is a coming-of-age story filled with humor, awkward moments, and a heartfelt exploration of friendship.",
    "the hangover": "'The Hangover' is a comedy film about three friends who retrace their steps after a wild bachelor party in Las Vegas to find their missing friend before his wedding. Known for its outrageous situations, memorable characters, and comedic chaos, the film was a huge success and spawned several sequels.",
    "step brothers": "'Step Brothers' is a comedy film about two middle-aged men, Brennan and Dale, who become stepbrothers when their single parents marry. The film explores their rivalry, antics, and eventual friendship. It is filled with humor, absurd situations, and the dynamic comedic performances of Will Ferrell and John C. Reilly.",
    "anchorman": "'Anchorman: The Legend of Ron Burgundy' is a comedy film about Ron Burgundy, a top-rated news anchor in 1970s San Diego. The story follows Burgundy and his news team as they face challenges posed by the arrival of a talented female reporter. The film is known for its satirical take on the news industry and its quirky characters.",
    "bridesmaids": "'Bridesmaids' is a comedy film that follows Annie, whose life unravels as she competes with her best friend's other bridesmaids for the coveted title of Maid of Honor. Known for its hilarious moments and heartwarming story, the film is praised for its ensemble cast and comedic timing.",
    "the notebook": "'The Notebook' is a romantic drama about a young couple, Noah and Allie, who fall in love during the early 1940s. The film chronicles their passionate romance, societal pressures, and enduring love despite various obstacles. It is celebrated for its emotional depth, beautiful storytelling, and the chemistry between its lead actors.",
    "pride and prejudice": "'Pride and Prejudice' is a romantic drama based on Jane Austen's novel. It follows the life of Elizabeth Bennet as she navigates issues of manners, morality, and marriage in early 19th century England. The film is known for its sharp wit, beautiful period costumes, and the timeless romance between Elizabeth and Mr. Darcy.",
    "la la land": "'La La Land' is a romantic musical film about an aspiring actress, Mia, and a jazz musician, Sebastian, as they pursue their dreams in Los Angeles. Directed by Damien Chazelle, the film is noted for its vibrant cinematography, memorable music, and a bittersweet love story that explores the sacrifices made for artistic ambitions.",
    "titanic": "'Titanic' is a romantic disaster film directed by James Cameron. It tells the fictionalized story of the RMS Titanic's ill-fated voyage and the romance between Jack, a poor artist, and Rose, a wealthy young woman. The film is known for its epic scale, visual effects, and the tragic love story that unfolds against the backdrop of the ship's sinking.",
    "before sunrise": "'Before Sunrise' is a romantic drama directed by Richard Linklater. It follows Jesse and Celine, who meet on a train and spend a night together in Vienna, sharing thoughts and experiences. The film is celebrated for its realistic dialogue, deep connection between characters, and the exploration of young love.",
    "inception": "'Inception' is a sci-fi thriller directed by Christopher Nolan. The film follows Dom Cobb, a thief who enters the dreams of others to steal secrets. Cobb is given a chance to have his criminal history erased if he can successfully plant an idea into someone's subconscious. Known for its complex narrative structure and stunning visual effects, 'Inception' explores themes of reality and dreams.",
    "interstellar": "'Interstellar' is a sci-fi film directed by Christopher Nolan. It follows a group of astronauts who travel through a wormhole near Saturn in search of a new habitable planet for humanity as Earth faces environmental collapse. The film is acclaimed for its scientific accuracy, emotional depth, and impressive visual effects.",
    "the matrix": "'The Matrix' is a sci-fi action film directed by the Wachowskis. It follows Neo, a computer hacker who discovers that reality as he knows it is a simulation created by sentient machines to subdue humanity. The film is renowned for its groundbreaking special effects, philosophical themes, and innovative action sequences.",
    "blade runner 2049": "'Blade Runner 2049' is a sci-fi film directed by Denis Villeneuve. It is a sequel to the 1982 film 'Blade Runner'. The story follows K, a replicant 'blade runner' who uncovers a secret that threatens to destabilize society. The film is noted for its stunning visual style, atmospheric world-building, and thought-provoking themes about humanity and artificial intelligence.",
    "guardians of the galaxy": "'Guardians of the Galaxy' is a sci-fi superhero film from Marvel. It follows Peter Quill and his ragtag group of intergalactic misfits as they band together to stop a powerful villain. The film is praised for its humor, vibrant characters, and an unforgettable soundtrack.",
    "the conjuring": "'The Conjuring' is a horror film directed by James Wan. It is based on the true story of paranormal investigators Ed and Lorraine Warren, who help a family terrorized by a dark presence in their farmhouse. The film is praised for its suspenseful atmosphere, effective scares, and strong performances.",
    "get out": "'Get Out' is a horror film written and directed by Jordan Peele. The film follows Chris, a young African American man who uncovers shocking secrets when he meets the family of his white girlfriend. 'Get Out' is acclaimed for its social commentary on race relations, its blend of horror and satire, and Peele's direction.",
    "hereditary": "'Hereditary' is a psychological horror film directed by Ari Aster. The story revolves around a family that begins to unravel following the death of their secretive grandmother. The film is known for its unsettling atmosphere, disturbing imagery, and powerful performances, particularly by Toni Collette.",
    "a quiet place": "'A Quiet Place' is a horror film directed by John Krasinski. Set in a post-apocalyptic world, it follows a family forced to live in silence while hiding from creatures that hunt by sound. The film is praised for its innovative use of sound, tension-filled narrative, and strong emotional core.",
    "the shawshank redemption": "'The Shawshank Redemption' is a drama film based on Stephen King's novella. It tells the story of Andy Dufresne, a man wrongfully imprisoned, and his friendship with fellow inmate Red. The film is celebrated for its powerful narrative, deep characters, and themes of hope and resilience.",
    "forrest gump": "'Forrest Gump' is a drama film starring Tom Hanks as a man with low intelligence who witnesses and unwittingly influences several historical events in 20th century America. The film is known for its emotional depth, compelling storytelling, and Hanks' memorable performance.",
    "the godfather": "'The Godfather' is a crime drama directed by Francis Ford Coppola, based on Mario Puzo's novel. It follows the powerful Italian-American crime family of Don Vito Corleone, focusing on the transformation of his youngest son, Michael, from reluctant family outsider to ruthless mafia boss. The film is acclaimed for its masterful direction, strong performances, and depiction of organized crime.",
    "fight club": "'Fight Club' is a drama film directed by David Fincher, based on Chuck Palahniuk's novel. It follows an unnamed narrator who forms an underground fight club with soap salesman Tyler Durden. The film explores themes of identity, consumerism, and rebellion, and is known for its dark humor and twist ending.",
    "a beautiful mind": "'A Beautiful Mind' is a biographical drama film about John Nash, a brilliant but asocial mathematician. Directed by Ron Howard, the film follows Nash's struggle with paranoid schizophrenia and his eventual triumph in the field of economics, for which he wins the Nobel Prize. The film is praised for its emotional depth and Russell Crowe's performance.",
    "toy story": "'Toy Story' is an animated film by Pixar that follows Woody, a pull-string cowboy doll, and Buzz Lightyear, a space ranger action figure, as they navigate the challenges of being toys in a world where they come to life when humans aren't around. The film is celebrated for its pioneering animation, heartwarming story, and memorable characters.",
    "spirited away": "'Spirited Away' is an animated fantasy film by Hayao Miyazaki. It follows Chihiro, a young girl who becomes trapped in a mysterious and magical world of spirits and must find a way to free herself and her parents. The film is renowned for its imaginative storytelling, stunning visuals, and rich cultural themes.",
    "finding nemo": "'Finding Nemo' is an animated adventure film by Pixar. It tells the story of Marlin, a clownfish, who embarks on a journey to find his missing son, Nemo. The film is known for its beautiful underwater animation, heartfelt narrative, and memorable characters.",
    "the lion king": "'The Lion King' is an animated musical film by Disney. It follows Simba, a young lion prince, who must embrace his destiny as the king of the Pride Lands after the death of his father, Mufasa. The film is beloved for its powerful story, iconic music, and emotional depth.",
    "up": "'Up' is an animated film by Pixar that follows Carl Fredricksen, an elderly man who attaches thousands of balloons to his house to fulfill his dream of seeing South America. Accompanied by a young boy named Russell, Carl discovers new adventures and learns to let go of his past. The film is praised for its emotional storytelling, humor, and vibrant animation."
}

def get_imdb_rating(movie_name):
    api_key = "bba2887f"  # OMDb API key
    url = f"http://www.omdbapi.com/?t={movie_name}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=5)  
        data = response.json()
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
    
    if results:
        return {"intent": classes[results[0][0]], "probability": str(results[0][1])}
    else:
        return None


#################################################################################
                            # Entity recognition
#################################################################################

def recognize_entities(text):
    print(f"Recognizing entities in text: {text}")
    doc = nlp(text)
    entities = {'movies': []}
    for ent in doc.ents:
        print(f"Found entity: {ent.text} with label: {ent.label_}")
        if ent.label_ == "WORK_OF_ART":  
            entities['movies'].append(ent.text.lower())  
            mentioned_movies.add(ent.text.lower())
    return entities['movies']



def extract_movie_name(sentence):
    movie_names_list = recognize_entities(sentence)  
    if not movie_names_list:
        for movie_name in movie_names:
            if movie_name.lower() in sentence.lower():
                movie_names_list.append(movie_name.lower())
    print(f"Extracted movie names: {movie_names_list}")
    for movie_name in movie_names_list:
        for key in movie_details.keys():
            if movie_name == key.lower():
                return key  # Return the correctly cased key from the dictionary
    print("No valid movie names found in sentence.")
    return ""



def update_last_recommended_movie(response):
    movie_name = extract_movie_name(response)
    if movie_name:
        global last_recommended_movie
        last_recommended_movie = movie_name
        print(f"Updated last recommended movie: {last_recommended_movie}")
    else:
        print("No movie name found to update last recommended movie.")

#################################################################################
                            # Sentiment analyser
#################################################################################
def analyze_sentiment(text, previous_sentiment=None):
    # VADER analysis
    sentiment_vader_scores = analyzer.polarity_scores(text)
    compound = sentiment_vader_scores['compound']
    if compound > 0.2:
        sentiment_vader = "positive"
    elif compound < -0.2:
        sentiment_vader = "negative"
    else:
        sentiment_vader = "neutral"

    # TextBlob analysis
    analysis = TextBlob(text)
    sentiment_tb_score = analysis.sentiment.polarity
    if sentiment_tb_score > 0.1:
        sentiment_tb = "positive"
    elif sentiment_tb_score < -0.1:
        sentiment_tb = "negative"
    else:
        sentiment_tb = "neutral"

    # Combining both
    if sentiment_vader == sentiment_tb:
        current_sentiment = sentiment_vader
    else:
        # If sentiments differ, combine both scores for a final decision
        combined_score = compound + sentiment_tb_score
        if combined_score > 0.1:
            current_sentiment = "positive"
        elif combined_score < -0.1:
            current_sentiment = "negative"
        else:
            current_sentiment = "neutral"

    if previous_sentiment:
        print(f"Went from {previous_sentiment} to {current_sentiment}")
    else:
        print(f"Sentiment detected: {current_sentiment}")

    return current_sentiment, sentiment_vader_scores


def get_response(intents_list, intents_json, user_query, previous_sentiment):
    global last_recommended_movie
    if not intents_list:
        return "I'm sorry, I don't understand.", previous_sentiment

    primary_intent = intents_list[0]  # Choose the most probable intent
    response_parts = []
    sentiment, sentiment_scores = analyze_sentiment(user_query, previous_sentiment)

    tag = primary_intent['intent']

    for i in intents_json['intents']:
        if i['tag'] == tag:
            print(f"Matching intent found: {tag}")
            if sentiment == "positive":
                response = random.choice(i.get('positive_responses', []))
            elif sentiment == "negative":
                response = random.choice(i.get('negative_responses', []))
            else:
                response = random.choice(i.get('neutral_responses', []))

            print(f"Generated response: {response}")

            # If this is a recommendation, update the last recommended movie
            if tag == "recommend_movie" or tag.startswith("recommend_"):
                update_last_recommended_movie(response)

            # If this is a request for movie details, use the last recommended movie
            if tag == "movie_details":
                print(f"User query: {user_query}")
                movie_name = extract_movie_name(user_query)
                if not movie_name and last_recommended_movie:
                    movie_name = last_recommended_movie
                movie_detail = movie_details.get(movie_name, f"I'm sorry, I don't have details about '{movie_name}'.")
                response = movie_detail
                print(f"Using movie name: {movie_name}")

            if tag == "imdb_rating":
                movie_name = extract_movie_name(user_query)
                if not movie_name and last_recommended_movie:
                    movie_name = last_recommended_movie
                rating, short_url = get_imdb_rating(movie_name)
                if short_url:
                    response = f"The IMDb rating for {movie_name} is {rating}. For more details, visit: {short_url}"
                else:
                    response = f"The IMDb rating for {movie_name} is {rating}."

            response_parts.append(response)
            break

    final_response = " ".join(response_parts)
    return final_response, sentiment_scores


def chatbot_response(msg, previous_sentiment):
    global last_recommended_movie
    global conversation_summary_generated
    intent = predict_class(msg, model)
    if not intent:
        return "I'm not sure how to respond to that.", previous_sentiment, None
    res, new_sentiment = get_response([intent], intents, msg, previous_sentiment)
    
    # Debugging statements to trace the process
    print(f"Input message: {msg}")
    print(f"Predicted intent: {intent}")
    print(f"Response: {res}")
    print(f"Last recommended movie before update: {last_recommended_movie}")

    if 'recommend_movie' in intent['intent'] or intent['intent'].startswith('recommend_'):
        update_last_recommended_movie(res)
        print(f"Updated last recommended movie in chatbot_response: {last_recommended_movie}")

    conversation_history.append((msg, res, new_sentiment))
    
    if msg.lower() == "quit" or 'farewells' in intent['intent']:
        summary = generate_summary(conversation_history)
        res += f"\n{summary}"

    return res, new_sentiment, last_recommended_movie



##################################################################################################################
##################################################################################################################

def generate_summary(conversation_history):
    discussed_movies = set(mentioned_movies)  
    sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
    word_freq = {}

    for entry in conversation_history:
        user_input, _, sentiment_scores = entry
        words = nltk.word_tokenize(user_input)
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
        for word in words:
            if word not in nltk.corpus.stopwords.words('english'):
                word_freq[word] = word_freq.get(word, 0) + 1

        sentiment = "positive" if sentiment_scores['compound'] > 0.6 else "negative" if sentiment_scores['compound'] < -0.6 else "neutral"
        sentiments[sentiment] += 1

    discussed_movies_list = list(discussed_movies)
    if discussed_movies_list:
        movie_summary = ", ".join(discussed_movies_list)
    else:
        movie_summary = "various movies"

    most_frequent_sentiment = max(sentiments, key=sentiments.get)
    word_freq_list = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    word_freq_str = ", ".join([f"{word}: {freq}" for word, freq in word_freq_list])
    summary = (f"In our conversation, we discussed {movie_summary}. "
               f"You felt {most_frequent_sentiment} through most of the conversation.\n"
               f"Most frequent words (excluding stopwords): {word_freq_str}.")
    return summary


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['POST'])
def send_message():
    message = request.json.get('message')
    previous_sentiment = request.json.get('previous_sentiment', None)
    response, new_sentiment, movie_name = chatbot_response(message, previous_sentiment)

    return jsonify({'response': response, 'sentiment': new_sentiment})

if __name__ == '__main__':
    app.run(debug=True)