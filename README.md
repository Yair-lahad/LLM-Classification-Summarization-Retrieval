# LLM-Classification-Summarization-Retrieval
Applied NLP. text classification, text summarization and information retrieval using Transformers (DistilBert), Tokenizaion, Pytorch and Tensorflow implementation

Part 1: Text Classification
<img width="848" height="796" alt="image" src="https://github.com/user-attachments/assets/6a588305-cfcf-4e27-9884-cdb0556c80ab" />
Emotion sentiment dataset is highly imbalanced - most examples are labeled as "neutral". A model that always predicts "neutral" could achieve high accuracy (around 80%) by exploiting the majority class, without actually learning to distinguish the minority classes. Precision and recall, on the other hand, help evaluate how well the model identifies less frequent emotions and whether it balances false positives and false negatives.

Sentiment Precision: 0.6708, Sentiment Recall: 0.4990 
<img width="755" height="580" alt="image" src="https://github.com/user-attachments/assets/d67384d2-d57e-4ced-8d69-8c1e78c1391b" />

Emotion Precision: 1.0000, Emotion Recall: 0.2130 
<img width="756" height="597" alt="image" src="https://github.com/user-attachments/assets/e923a52f-24b8-4926-ad20-38cf042a5f5b" />

Yes, there is a significant drop in performance.
-  Sentiment dataset performance dropped because the model overfit to predicting "neutral", likely due to class imbalance in the training data. This caused it to miss many Non-Neutral cases.
- The Emotion dataset contains only Non-Neutral examples, so while the model achieved perfect precision (it was correct every time it predicted Non-Neutral), it missed most true cases -showing poor generalization.

Part 2: Text Summarization
<img width="1170" height="499" alt="image" src="https://github.com/user-attachments/assets/3fa8842c-a916-4e1d-9575-49416e4dacbf" />

Type of ROUGE	Max Score	Min Score
 ROUGE-1 (unigrams)	1	0.2
 ROUGE-2 (bigrams)	0.888	0
 
Analyzing smallest ROUGE-2 score: 0. Reference summary is completely abstractive and not extractive, not a single bigram (2 words) intersection between reference (highlights) and candidate (article text). Abstraction seems reasonable because the current article includes many names, numbers and redundant info which can be summarized as "huge name", etc. Explaining the imporatnce of the company without the need to specify exect reference

<img width="853" height="458" alt="image" src="https://github.com/user-attachments/assets/ce52eb34-08f5-44a9-8e6e-06554288eb43" />


Part 3 - Information Retrieval

Most similar article to the following query from 1000 first rows:

Query: Leonardo DiCaprio
Most Relevant Article: Elizabeth was portrayed in a variety of media by many notable artists, including painters Pietro Annigoni, Peter Blake, Chinwe Chukwuogo-Roy, Terence Cuneo, Lucian Freud, Rolf Harris, Damien Hirst, Juliet Pannett and Tai-Shan Schierenberg. Notable photographers of Elizabeth included Cecil Beaton, Yousuf Karsh, Anwar Hussein, Annie Leibovitz, Lord Lichfield, Terry O'Neill, John Swannell and Dorothy Wilding. The first official portrait photograph of Elizabeth was taken by Marcus Adams in 1926.     
Similarity Score: 0.5368194

Query: France
Most Relevant Article: In May 2022, FIFA announced the list of 36 referees, 69 assistant referees, and 24 video assistant referees for the tournament. Of the 36 referees, FIFA included two each from Argentina, Brazil, England, and France.
Similarity Score: 0.36249438

Query: Python
Most Relevant Article: SharePoint, a web collaboration platform codenamed as Office Server, has integration and compatibility with Office 2003 and so on.
Similarity Score: 0.5572224

Query: Deep Learning
Most Relevant Article: SharePoint, a web collaboration platform codenamed as Office Server, has integration and compatibility with Office 2003 and so on.
Similarity Score: 0.56057405

Conclusion:
The results are weak because the queries are very short and ambiguous (one word) and the model uses mean pooling, which averages out important context. Combined with scanning only the first 1000 random Wikipedia entries, most are unrelated. the system finds the most similar article, but that similarity is still low and often not meaningful.

