import nltk
import spacy
import gensim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Tokenisasi dengan NLTK dan spaCy
nltk.download('punkt')
text = "Ini adalah contoh kalimat untuk analisis teks bahasa Indonesia."
tokens_nltk = nltk.word_tokenize(text)

nlp = spacy.blank("id")
doc = nlp(text)
tokens_spacy = [token.text for token in doc]

# Word2Vec
corpus = [["produk", "ini", "sangat", "bagus"], ["sangat", "buruk", "dan", "mengecewakan"]]
model = gensim.models.Word2Vec(sentences=corpus, vector_size=50, window=2, min_count=1)

# Visualisasi Word Embeddings
words = list(model.wv.index_to_key)
vectors = [model.wv[w] for w in words]
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.title("PCA Word Embeddings")
plt.savefig("word_embeddings_pca.png")