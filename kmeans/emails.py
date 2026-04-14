import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

file_path = "/mydata/emails.csv"  
data = pd.read_csv(file_path)

print(data.head())

def extract_body(message):
    if pd.isnull(message):
        return ""
    split_message = message.split("\n\n", 1)  
    return split_message[1] if len(split_message) > 1 else ""

data['EmailBody'] = data['message'].apply(extract_body)

vectorizer = TfidfVectorizer(stop_words='english', max_features=500) 
X = vectorizer.fit_transform(data['EmailBody'])



num_clusters = 5  
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)


data['Cluster'] = kmeans.labels_


print(data['Cluster'].value_counts())

pca = PCA(n_components=2, random_state=42)
X_reduced = pca.fit_transform(X.toarray())