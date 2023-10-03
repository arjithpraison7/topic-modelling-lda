import requests
from bs4 import BeautifulSoup
from gensim import corpora, models
import gensim.corpora as corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis


# Data Collection
url = 'https://www.bbc.com/news/world'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
articles = soup.find_all('div', class_='gs-c-promo-body')  # Updated class name

# Data Preprocessing
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under', 'again', 'further', 'then', 'once'
])

processed_articles = []

for article in articles:
    text = article.text.lower()  # Extract the text content of the article
    tokens = [token for token in text.split() if token.isalpha() and token not in stop_words]
    processed_articles.append(tokens)

# Check if there are any valid tokens
if not any(processed_articles):
    raise ValueError("No valid tokens found in the processed articles. Please check your preprocessing steps.")

# Assuming 'processed_articles' is a list of tokenized and preprocessed articles
dictionary = corpora.Dictionary(processed_articles)
corpus = [dictionary.doc2bow(text) for text in processed_articles]

# Fit LDA model
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Print the topics and associated words
for topic_id, topic_words in lda_model.print_topics():
    print(f'Topic {topic_id}: {topic_words}')

dictionary = corpora.Dictionary(processed_articles)

# Create the Document-Term Matrix
doc_term_matrix = [dictionary.doc2bow(text) for text in processed_articles]

# Print an example of the Document-Term Matrix
print(doc_term_matrix[0])  # Print the DTM for the first document

# Assuming 'doc_term_matrix' is the Document-Term Matrix
lda_model = LdaModel(corpus=doc_term_matrix, id2word=dictionary, num_topics=5, passes=15)

for i, document in enumerate(doc_term_matrix):
    topic_distribution = lda_model.get_document_topics(document)
    print(f"Document {i}:")
    for topic, prob in topic_distribution:
        print(f"Topic {topic}: {prob:.4f}")

vis_data = gensimvis.prepare(lda_model, doc_term_matrix, dictionary)

# Save the visualization to an HTML file
pyLDAvis.save_html(vis_data, 'topic_modeling_visualization.html')
