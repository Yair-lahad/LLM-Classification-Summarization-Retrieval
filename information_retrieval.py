from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# Initialize tokenizer and model for DistilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = TFAutoModel.from_pretrained("distilbert-base-multilingual-cased")


def compute_embedding(text):
    encoded_input = tokenizer(
        text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(**encoded_input)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    return embeddings.numpy()


# Load a subset of the wikipedia dataset (assuming structure and availability)
dataset = load_dataset(
    "Cohere/wikipedia-22-12-en-embeddings", split="train", streaming=True)

# ========Exercise 3.1 ===========
def find_most_relevant_article(query_embedding, dataset, max_num_of_articles=None):
    max_similarity = -1
    most_relevant_article = None

    for i, article in enumerate(tqdm(dataset)):
        if max_num_of_articles and i >= max_num_of_articles:
            break
        try:
            article_text = article.get("text") or article.get(
                "content") or article.get("title")
            if not article_text:
                continue
            article_embedding = compute_embedding(article_text)
            similarity = cosine_similarity(
                query_embedding, article_embedding)[0][0]

            if similarity > max_similarity:
                max_similarity = similarity
                most_relevant_article = article_text
        except Exception:
            continue

    return most_relevant_article, max_similarity


# ========Exercise 3.2 ===========
queries = ["Leonardo DiCaprio", "France", "Python", "Deep Learning"]
articlesMaxNum = 1000
for q in queries:
    print(f"\nQuery: {q}")
    inputEmbedding = compute_embedding(q)
    article, similarity = find_most_relevant_article(
        inputEmbedding, dataset, articlesMaxNum)
    print("Most Relevant Article:", article)
    print("Similarity Score:", similarity)
