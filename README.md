
#### Marathi to Marathi Q&A System ####


### CodeBase
`Link -  https://colab.research.google.com/drive/1VfwJ13KtNbJVmLrA2gU-hWFhh-ho9BqP?usp=sharing`



## Prerequisites

Before running the project, ensure you have installed the required Python libraries:

```bash
pip install googletrans==4.0.0-rc1
pip install jsons
pip install deep-translator
pip install pandas
pip install stanza
pip install spacy
pip install transformers
pip install torch
```

## Instructions

### 1. Translation from English to Marathi

- Load the English dataset from KAGGLE-QUAC (`Aug_English.json`) and translate the relevant fields (`context`, `question`, and `answer_text`) into Marathi.
- Save the translated data into a new JSON file named `Aug_Marathi.json`.

### 2. Convert Translated JSON Data to CSV

- After the translation step, convert the translated JSON file (`Aug_Marathi.json`) into a CSV format and save it as `Aug_Marathi.csv`.

### 3. Process the Original Dataset

- Load the MAHASQUAD Marathi dataset (`train.json`).
- Extract and clean the relevant fields (ID, title, context, question, and answer text).
- Save the cleaned dataset as `main_traindata_set.csv`.

### 4. Clean and Format Data

- Remove unwanted punctuation and characters from both the original and translated datasets.
- Ensure that empty or null fields are cleaned, especially in the `answers_text` column, to maintain data quality.

### 5. Combine the Datasets

- Combine the cleaned original dataset (`main_traindata_set.csv`) and the translated dataset (`Aug_Marathi.csv`) into a single comprehensive dataset.
- Save this unified dataset as `Ultimate_data.csv`.

### 6. Final Output

The final combined dataset is saved as `Ultimate_data.csv`, which contains both the original Marathi data and the translated Marathi data, ready for further use in any NLP tasks.

## Files Generated

- **Aug_Marathi.json**: Contains the Marathi-translated dataset.
- **Aug_Marathi.csv**: Translated dataset in CSV format.
- **main_traindata_set.csv**: Cleaned original English dataset.
- **Ultimate_data.csv**: Final combined dataset with both English and Marathi data.



### Tokenization

- Followed word based Tokenization for tokenizing the data
- Get the unique word from all rows and store them in `tokens` set.
- Handled one problem occuring with date , the whole date with month should be considered as one token.
- Above Problem is done by writing `Custom two pointers algorithm`.

### Stop Word Removal

- Found out the most commonly used `Stop words in Marathi`.
- And also some stop words from `Github`.
- Combined both and made on Stop Word list
- Removed all those `Stop words` from the `Tokens` set.

### Stemming
The goal of the stemming system is to reduce words in Marathi to their base or root forms by removing suffixes, improving text processing tasks like Question Answering (QA) with context in Marathi.

- **Features**:
- **Rule-based Stemming**:
A manual stemming process is implemented using predefined suffix stripping rules for `complex suffixes`, `join word suffixes`, and `inflectional suffixes` (for consonants like 'च', 'ल', etc.).
This approach involves eliminating suffixes based on `predefined linguistic rules for Marathi`.

-**Unsupervised Stemming (Statistical Approach)**:
This approach uses an `n-gram splitting method` to identify possible stems and suffixes.
Suffixes are statistically evaluated based on their `frequency`, and rules for suffix stripping are generated dynamically.
Instructions

1. **Rule-Based Stemming**
Files/Classes:

RecursiveMarathiStemmer: This class implements the rule-based stemming process using recursive stripping of predefined suffixes.
Steps:

Initialize the stemmer:

python
Copy code
stemmer = RecursiveMarathiStemmer()
Stem individual words or sentences:

python
Copy code
stemmed_word = stemmer.stem_word("मुलगी")
print(f"Stemmed Word: {stemmed_word}")

stemmed_sentence = stemmer.stem_sentence("राजा मुलीच्या घरी गेला")
print(f"Stemmed Sentence: {stemmed_sentence}")
Customize Suffix Rules:

Modify the complex_suffixes, join_word_suffixes, ch_suffixes, l_suffixes, and plain_suffixes lists within the class for more customized stripping behavior.


2. **Unsupervised Statistical Stemming**
Files/Classes:

UnsupervisedStemmer: This class implements an unsupervised statistical stemming method based on n-grams and frequency-based suffix stripping.
Steps:

Initialize the stemmer:

python
Copy code
from collections import defaultdict
stemmer = UnsupervisedStemmer()
Add words to stem classes:

The model learns stem classes by splitting words into n-grams and analyzing suffix patterns.
python
Copy code
tokens = {'अंकात', 'अंका', 'अंके', 'अंकेला', 'अंक'}
stemmer.generate_stem_classes(tokens)
Generate suffix rules:

The statistical rules for suffix stripping are generated based on the frequency of suffixes in the corpus.
python
Copy code
stem_suffixes = stemmer.generate_suffix_list()
statistical_rules = stemmer.generate_suffix_rules_statistical(stem_suffixes, total_words=len(tokens))
Apply suffix stripping:

Strip suffixes from the words using the generated rules:
python
Copy code
stemmed_tokens = set()
for token in tokens:
    stemmed_token = stemmer.apply_suffix_rules(token, statistical_rules)
    stemmed_tokens.add(stemmed_token)

print(f"Original Tokens: {tokens}")
print(f"Stemmed Tokens: {stemmed_tokens}")



### POS Tagging Using Stanza

- In the process of performing `lemmatization` for Marathi words, it is essential to have `Part-of-Speech (POS)` labels for each tokenized word. To achieve this, we use Stanza, a robust natural language ----processing (NLP) library developed by Stanford, designed specifically for various NLP tasks, including POS tagging.

**Steps**:
- Initialize Stanza:
The Stanza pipeline is initialized for Marathi and stored in the nlp variable. In the function `pos_tagging_stanza_for_tokens`, an empty list `tagged_tokens` is created to store tokens with their respective POS labels.


nlp = stanza.Pipeline('mr', processors='pos', use_gpu=True)

**Token Processing**:
The tokens are stored in a set called `Five_tokens`, which is passed as a parameter to the function. The function processes each token and returns a list of words with their corresponding POS labels, which is stored in the tagged variable.


tagged = pos_tagging_stanza_for_tokens(Five_tokens)
This is just a part of the larger code, primarily focused on POS tagging.



### Lemmatization

- Once we have the POS tags for the Marathi tokens (stored in the tagged variable), we use them to perform `lemmatization`.

**Key Points**:
Handling Verb Exceptions:
For certain verb exceptions, we define a dictionary that maps specific past tense forms of verbs to their base form (infinitive). For instance:

verb_exceptions = {
    'आला': 'येणे',  # 'Came' -> 'To Come'
    'गेला': 'जाणे',  # 'Went' -> 'To Go'
}
 

When we encounter tokens like `आला`, we replace it with `येणे`, similar to replacing `came` with `come` in English.

**Verb Suffixes**:
- We handle verb suffixes by removing endings such as `त आहे` (present continuous suffixes) and replacing them with `णे`, converting verbs to their infinitive form.

**Pronoun Mapping**:
- For pronouns, we map words like ` "he", "him", "his", etc., ` to their base forms.

**Lemmatization Process**:
- We pass the tagged list to the lemmatize function, which checks the POS tag of each word. If it’s a verb or a pronoun, it applies the above rules and returns the token, its lemma, and its POS tag.



### Embedding Space

- After lemmatization, we `extract only the lemmas` (since they are in their base form), and use the Word2Vec library to convert these lemmas into numeric embeddings, which are required for machine learning models.

Steps:
Initialize Word2Vec Model:
We use Word2Vec to create an embedding model from the list of lemmas. The key parameters are:

vector_size=100: The size of the vector representation for each word.
window=1: It considers the neighbors (left and right) of the target word while generating embeddings.
min_count=1: Words that appear less than this count are ignored.
workers=4: Specifies the number of CPU threads for training.

model = Word2Vec(sentences, vector_size=100, window=1, min_count=1, workers=4)
Saving the Model:
We save the trained Word2Vec model to a file (word2vec_model.bin) for future use.

Storing Embeddings:
The resulting embeddings are stored in a dictionary where the key is the lemma and the value is the corresponding embedding.

embeddings = {lemma: model.wv[lemma] for lemma in lemmas}




### Datasets
`Link1 -  https://github.com/l3cube-pune/MarathiNLP/blob/main/L3Cube-MahaSQuAD/train.json.zip `

`Link2 - https://www.kaggle.com/datasets/jeromeblanchet/quac-question-answering-in-context-dataset`

