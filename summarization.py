import matplotlib.pyplot as plt
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline

# Download necessary resources from nltk. commeting out to avoid repeated downloads

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')

dataset = load_dataset("cnn_dailymail", '3.0.0')
df = dataset['train'].to_pandas()
df = df.head(1000)


def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    # Convert to lower case
    tokens = [token.lower() for token in tokens]
    # Remove stop words (optional)
    stop_words = set(stopwords.words('english'))
    tokens = [
        token for token in tokens if token not in stop_words and token.isalpha()]
    return " ".join(tokens)


# ====== Part 2.1 =====================
def add_length_columns(df):
    # Calculate lengths of article and highlights
    df['article_len'] = df['article'].apply(
        lambda x: len(preprocess_text(x).split()))
    df['highlights_len'] = df['highlights'].apply(
        lambda x: len(preprocess_text(x).split()))
    return df


add_length_columns(df)
# ====== Part 2.2 =====================


def set_title_labels(ax, title=None, xlabel=None, ylabel='Count'):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_hist_on_ax(ax, df, col_name, bins=30, color='skyblue', show_mean=True):
    # Plots a histogram of df[col_name] on the given subplot ax.
    ax.hist(df[col_name], bins=bins, color=color, edgecolor='black')
    if show_mean:
        mean_val = df[col_name].mean()
        ax.axvline(mean_val, color='green', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Mean: {mean_val:.1f}')
        ax.legend()
    set_title_labels(ax, title=f'Histogram of {col_name}', xlabel=col_name)


# Plotting histograms for article and highlights lengths - uncomment to visualize

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_hist_on_ax(axes[0], df, 'article_len', bins=30, color='salmon')
plot_hist_on_ax(axes[1], df, 'highlights_len', bins=30)
plt.tight_layout()
plt.show()

# ======Part 2.3 ================


def ngrams(text, n):
    # Generate N-grams from the preprocessed text
    processed_text = preprocess_text(text)  # Preprocess the text first
    words = processed_text.split()
    return set(zip(*[words[i:] for i in range(n)]))


def rouge_n(reference, candidate, n):
    # Calculate the N-gram overlap between reference and candidate
    ref_ngrams = ngrams(reference, n)
    cand_ngrams = ngrams(candidate, n)
    if not ref_ngrams:
        return 0.0
    overlap = ref_ngrams.intersection(cand_ngrams)
    return len(overlap) / len(ref_ngrams)


# Example of calculating Rouge-1 and Rouge-2 for a dataframe
df['rouge_1'] = df.apply(lambda row: rouge_n(
    row['highlights'], row['article'], 1), axis=1)
df['rouge_2'] = df.apply(lambda row: rouge_n(
    row['highlights'], row['article'], 2), axis=1)

isUni = False  # Set to True for Rouge-1 (Unigram), False for Rouge-2 (Bigram)
isMin = False  # Set to True for minimum, False for maximum
curr_n_col = 'rouge_1' if isUni else 'rouge_2'
headline = 'Smallest' if isMin else 'Highest'
plt.figure(figsize=(12, 6))
plt.hist(df[curr_n_col], bins=30, color='blue', alpha=0.7)
plt.title(f'{curr_n_col} score distribution on ground truth')
plt.show()

max_rouge_index = df[curr_n_col].argmax()
min_rouge_index = df[curr_n_col].argmin()
curr_ind = min_rouge_index if isMin else max_rouge_index
print(f"at index- {min_rouge_index} Minimum {curr_n_col} score:",
      df[curr_n_col].min())
print(f"at index- {max_rouge_index} Maximum {curr_n_col} score:",
      df[curr_n_col].max())
print("========================\n")
print(f"Article with {headline} {curr_n_col} score:",
      df.iloc[curr_ind]['article'])
print("========================\n\n")
print(f"Highlights with {headline} {curr_n_col} score:",
      df.iloc[curr_ind]['highlights'])


# =========== 2.4 ================
# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")


def summarize_text(text):
    # Summarizing the text using the pipeline
    summary = summarizer(text, max_length=40, min_length=5, do_sample=False)
    print("-")
    return summary[0]['summary_text']


# Calculate the rouge-2 score of the first 10 entries
df10 = df.iloc[:10].copy()  # Extract only the first 10 rows for this section
# Generate summaries
df10['generated_summary'] = df10['article'].apply(summarize_text)
# Compute ROUGE-2 between highlights and generated summary
df10['generated_rouge_2'] = df10.apply(
    lambda row: rouge_n(row['highlights'], row['generated_summary'], 2),
    axis=1
)
# Compare with original article ROUGE-2 (computed earlier in df['rouge_2'])

# bring in existing baseline values
df10['rouge_2'] = df.loc[:9, 'rouge_2'].values
df10['model_worse'] = df10['generated_rouge_2'] < df10['rouge_2']
# Print for Word-compatible table
for idx, row in df10.iterrows():
    print(f"ID {idx+1}")
    # print(f"Article (Summary): {row['article']}")
    print(f"Reference (Highlight): {row['highlights']}")
    print(f"Generated Summary   : {row['generated_summary']}")
    print(f"ROUGE-2 (Generated) : {row['generated_rouge_2']:.4f}")
    print(f"ROUGE-2 (Article)   : {row['rouge_2']:.4f}")
    print(f"Model Worse?        : {'Yes' if row['model_worse'] else 'No'}")
    print("-" * 60)
