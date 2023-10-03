# Topic Modelling on Text Data

# Description

### Project Overview:

The "Topic Modeling on News Articles" project aims to utilize the powerful technique of Latent Dirichlet Allocation (LDA) to uncover hidden themes within a collection of news articles. This unsupervised learning approach will allow us to identify distinct topics, providing valuable insights into the underlying content of the articles.

### Objectives:

- Apply Latent Dirichlet Allocation (LDA) to perform topic modeling on a diverse set of news articles.
- Create a document-term matrix to represent the frequency of words in each document.
- Explore and analyze the identified topics to gain a deeper understanding of the content.

# Libraries used

For this project on Topic Modeling using Latent Dirichlet Allocation (LDA), you'll primarily be working with Python and some popular libraries. Here are the key libraries you'll need:

1. **NLTK (Natural Language Toolkit)**:
    - Purpose: NLTK provides essential tools and resources for working with human language data (text).
    - Functions: Tokenization, stopwords removal, lemmatization, and more.
    - Website: **[NLTK Official Website](https://www.nltk.org/)**
2. **Gensim**:
    - Purpose: Gensim is a robust library for natural language processing and topic modeling. It provides an implementation of LDA.
    - Functions: LDA modeling, document-term matrix creation, and more.
    - Website: **[Gensim Official Website](https://radimrehurek.com/gensim/)**
3. **Scikit-Learn (sklearn)**:
    - Purpose: Scikit-Learn is a powerful library for machine learning in Python. It also provides an LDA implementation.
    - Functions: LDA modeling, evaluation metrics, and more.
    - Website: **[Scikit-Learn Official Website](https://scikit-learn.org/stable/)**
4. **Pandas**:
    - Purpose: Pandas is a versatile data manipulation library that helps in handling data structures efficiently.
    - Functions: Data preprocessing, data manipulation, and analysis.
    - Website: **[Pandas Official Website](https://pandas.pydata.org/)**
5. **Matplotlib and Seaborn**:
    - Purpose: These libraries are used for data visualization in Python.
    - Functions: Plotting graphs, creating visualizations for topics, etc.
    - Websites: **[Matplotlib Official Website](https://matplotlib.org/)**, **[Seaborn Official Website](https://seaborn.pydata.org/)**
6. **NumPy**:
    - Purpose: NumPy is a fundamental package for numerical computations in Python.
    - Functions: Working with arrays, matrices, and mathematical functions.
    - Website: **[NumPy Official Website](https://numpy.org/)**
7. **WordCloud**:
    - Purpose: This library is used to create word clouds, which can be a visual representation of important words in your documents.
    - Website: **[WordCloud on GitHub](https://github.com/amueller/word_cloud)**
8. **Matplotlib for WordCloud**:
    - Purpose: To visualize the generated word clouds.
    - Website: **[Matplotlib Official Website](https://matplotlib.org/)**

# Steps

1. **Data Collection and Preprocessing**
2. **Document-Term Matrix Creation**
3. **Application of Latent Dirichlet Allocation (LDA)**
4. **Interpretation of Results**
5. **Visualization and Analysis**
6. **Evaluation and Fine-Tuning**
7. **Expected Outcomes**
8. **Learning Goals**

### Data Collection:

- **Objective**: Gather a comprehensive dataset of news articles from diverse sources and domains.

### **Tasks:**

1. **Source Selection**: Identify reliable sources for news articles. This could include reputable news websites, blogs, or specialized news APIs.
2. **Data Gathering Method**:
    - If using a public dataset, download and import it into your project.
    - If scraping, consider using Python libraries like **`requests`** and **`BeautifulSoup`** for web scraping. Be sure to respect the terms of service of the websites you scrape.
3. **Data Organization**: Store the collected data in an organized format, such as a structured dataset or a directory with individual text files.

### Data Preprocessing:

- **Objective**: Prepare the collected text data for analysis by applying necessary transformations.

### **Tasks:**

1. **Text Cleaning**:
    - Remove any HTML tags, special characters, or non-alphabetic characters.
    - Address any specific data quirks or anomalies you may have encountered during data collection.
2. **Tokenization**:
    - Break the text into individual words or tokens. This will be the basis for further analysis.
3. **Lowercasing**:
    - Convert all text to lowercase. This ensures uniformity in word frequencies.
4. **Stopword Removal**:
    - Remove common words (e.g., "the", "and", "is") that do not provide much information about the content.
5. **Lemmatization or Stemming**:
    - Reduce words to their base or root form to further normalize the text. Choose between lemmatization (contextually aware) or stemming (rule-based).
6. **Document Organization**:
    - If applicable, organize the data into a structured format (e.g., a pandas DataFrame) for ease of manipulation.

```python
import requests
from bs4 import BeautifulSoup
from gensim import corpora, models

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
```

For this project we have chosen to use the BBC World News Website - [https://www.bbc.com/news/world](https://www.bbc.com/news/world)

```python
Topic 0: 0.035*"oil" + 0.028*"says" + 0.028*"church" + 0.021*"hours" + 0.021*"video" + 0.021*"high" + 0.021*"cartel" + 0.021*"prolonged" + 0.021*"warns" + 0.021*"leader"
Topic 1: 0.036*"hours" + 0.013*"worklife" + 0.013*"why" + 0.013*"travel" + 0.013*"fraud" + 0.013*"blast" + 0.013*"trump" + 0.013*"court" + 0.013*"turns" + 0.013*"canada"
Topic 2: 0.069*"speaker" + 0.052*"house" + 0.052*"bid" + 0.035*"historic" + 0.035*"us" + 0.035*"oust" + 0.030*"hours" + 0.018*"early" + 0.018*"begins" + 0.018*"voting"
Topic 3: 0.038*"hours" + 0.038*"pledge" + 0.038*"women" + 0.020*"says" + 0.020*"there" + 0.020*"leave" + 0.020*"despite" + 0.020*"burger" + 0.020*"exitthe" + 0.020*"new"
Topic 4: 0.050*"shares" + 0.026*"agobusiness" + 0.026*"suspended" + 0.026*"trading" + 0.026*"embattled" + 0.026*"evergrande" + 0.026*"market" + 0.026*"jump" + 0.026*"chinese" + 0.026*"minutes"
```

Here's a brief summary of the identified topics:

**Topic 0:**

- Keywords: oil, says, church, hours, video, high, cartel, prolonged, warns, leader

**Topic 1:**

- Keywords: hours, worklife, why, travel, fraud, blast, trump, court, turns, canada

**Topic 2:**

- Keywords: speaker, house, bid, historic, us, oust, hours, early, begins, voting

**Topic 3:**

- Keywords: hours, pledge, women, says, there, leave, despite, burger, exitthe, new

**Topic 4:**

- Keywords: shares, agobusiness, suspended, trading, embattled, evergrande, market, jump, chinese, minutes

Each topic represents a set of keywords that are most relevant to that topic.

## Document term matrix creation

Document-Term Matrix (DTM) creation, involves converting the processed text data into a structured numerical format that can be used for further analysis.

In the context of topic modeling, a Document-Term Matrix is a matrix representation where rows correspond to documents (in this case, articles) and columns correspond to terms (words). Each cell in the matrix represents the frequency of a particular term in a specific document.

To create the Document-Term Matrix, we'll use the **`corpora.Dictionary`** object and the list of processed articles.

Here's how you can do it:

```python
import gensim.corpora as corpora

dictionary = corpora.Dictionary(processed_articles)

# Create the Document-Term Matrix
doc_term_matrix = [dictionary.doc2bow(text) for text in processed_articles]

# Print an example of the Document-Term Matrix
print(doc_term_matrix[0])  # Print the DTM for the first document
```

In this code, we first create a **`Dictionary`** object from the processed articles. Then, we use this dictionary to create the Document-Term Matrix (**`doc_term_matrix`**). Finally, we print an example of the Document-Term Matrix for the first document.

This matrix will serve as the basis for training the topic modeling algorithm. If you have any further questions or if you'd like to proceed with the next step, feel free to let me know!

```python
[(0, 1), (1, 3), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 3), (9, 1), (10, 1), (11, 1), (12, 2), (13, 1), (14, 1), (15, 1), (16, 1), (17, 4), (18, 2)]
```

Each tuple in the list represents a term in the format **`(term_id, term_frequency)`**. For example, **`(0, 1)`** means that term with ID 0 (which corresponds to a specific word) occurs once in the document.

## **Application of Latent Dirichlet Allocation (LDA)**

The next step in the topic modeling process is to train the LDA model. This involves using the Document-Term Matrix (DTM) that we created earlier to find the underlying topics in the text data.

Here's how you can train the LDA model:

```python
from gensim.models import LdaModel

lda_model = LdaModel(corpus=doc_term_matrix, id2word=dictionary, num_topics=5, passes=15)
```

In this code, we're using the **`LdaModel`** class to train the model. We're specifying that we want to find 5 topics (**`num_topics=5`**) and we're running 15 iterations (**`passes=15`**) to fine-tune the topics.

### Assigning topics to documents

To assign topics to documents using the trained LDA model, we can use the **`get_document_topics()`** method provided by Gensim. This method takes a document represented as a bag-of-words vector (in the form of a list of tuples) and returns the distribution of topics in that document.

Here's how you can do it:

```python

for i, document in enumerate(doc_term_matrix):
    topic_distribution = lda_model.get_document_topics(document)
    print(f"Document {i}:")
    for topic, prob in topic_distribution:
        print(f"Topic {topic}: {prob:.4f}")
```

In this code, we iterate through each document represented as a bag-of-words vector. For each document, we use **`lda_model.get_document_topics()`** to obtain the distribution of topics. The result is a list of tuples, where each tuple contains a topic ID and the corresponding probability of that topic in the document.

By printing out the topic distribution for each document, you'll be able to see which topics are prevalent in each document.

## Visualisation and Analysis

### Visualisation Technique

One of the popular libraries for interactive visualization of topic models is **`pyLDAvis`**. It provides an interactive web-based visualization that allows you to explore topics, their prevalence, and the most relevant terms.

Here's how you can use **`pyLDAvis`** with your trained LDA model:

```python
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

vis_data = gensimvis.prepare(lda_model, doc_term_matrix, dictionary)

# Save the visualization to an HTML file
pyLDAvis.save_html(vis_data, 'topic_modeling_visualization.html')
```

This will save the interactive visualization to a file named **`topic_modeling_visualization.html`** in your current directory. You can then open this HTML file in a web browser to explore the visualization.

### Analysis Technique

Analyzing the document-topic distribution is a valuable step in understanding how topics are distributed across your entire corpus. This can provide insights into which topics are prevalent and how documents are associated with different topics.

```python
for i, document in enumerate(doc_term_matrix):
    topic_distribution = lda_model.get_document_topics(document)
    print(f"Document {i}:")
    for topic, prob in topic_distribution:
        print(f"Topic {topic}: {prob:.4f}")
```

## Evaluation and Fine Tuning

### Fine Tuning

Fine-tuning involves making adjustments to the topic modeling process to improve the quality of the generated topics.

```python
# Example: Changing the number of topics
lda_model = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)
```

## **Learning Outcome:**

By working on this project on Topic Modeling with Latent Dirichlet Allocation (LDA), you will gain several valuable skills and insights:

1. **Text Preprocessing Techniques**: You'll become proficient in techniques like tokenization, stopword removal, lemmatization, and more, which are crucial for preparing text data for analysis.
2. **Unsupervised Learning with LDA**: You'll learn how to apply unsupervised learning techniques to identify hidden topics within a collection of documents.
3. **Data Collection and Scraping**: If you decide to scrape data from websites, you'll learn how to extract relevant information from web pages.
4. **Data Visualization and Analysis**: You'll gain experience in visualizing and analyzing topics using techniques like word clouds, bar plots, and interactive visualizations.
5. **Model Evaluation and Fine-Tuning**: You'll learn how to assess the quality of your topic model using metrics like coherence scores and perplexity. You'll also gain insights into fine-tuning the model for better results.
6. **Topic Evolution (Optional)**: If your dataset has a temporal component, you'll learn how to analyze how topics evolve over time.

## **Conclusion:**

In conclusion, this project provides a hands-on experience in applying topic modeling techniques to uncover hidden patterns and themes within text data. You'll be equipped with valuable skills in text processing, unsupervised learning, and model evaluation. Additionally, you'll have the ability to draw meaningful insights from the topics generated, which can be applied to various domains such as content analysis, customer feedback analysis, and more.

This project serves as a solid foundation for further exploration into natural language processing (NLP) and text analytics. It also demonstrates the power of unsupervised learning in uncovering valuable information from unstructured text data.

# Author

[Arjith Praison](https://www.linkedin.com/in/arjith-praison-95b145184/)

University of Siegen
