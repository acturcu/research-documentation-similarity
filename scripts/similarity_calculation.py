import os
import time

from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
import seaborn as sns
import numpy as np
import scipy.sparse as sp

files_dictionary = {}
repo_clusters = {}
num_clusters = 4
licenses = []
urls = []
technical_terms = ['parse', 'print', 'debug', 'exception', 'error', 'warning', 'recognition', 'detection', 'test', 'mod',
                   'game', 'render', ]
accuracy_list = []

def read_processed_file(location):
    content = []
    with open(location, 'r', encoding='utf-8') as file:
        content.append(file.read())
    return ' '.join(content)


# urls and licenses are not passed anymore
def read_processed_folder(directory, files):
    content = []
    for file in files:
        if file in files_dictionary[directory]:
            location = os.path.join("../processedDocumentation", directory, file)
            content.append(read_processed_file(location))
        else:
            content.append('')
    return ' '.join(content)

# Increase the weight of selected words in the TF-IDF vector
def modify_weights(doc):
    return {word: tfidf * 1.5 if word in technical_terms else tfidf for word, tfidf in doc.items()}


def get_similarity_matrix(files):
    content = [read_processed_folder(folder, files) for folder in files_dictionary.keys()]

    docs = len(content)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(content)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    custom_tfidf_matrix = sp.lil_matrix(tfidf_matrix.shape, dtype=tfidf_matrix.dtype)
    for i in range(docs):
        custom_scores = modify_weights({feature_names[j]: tfidf_matrix[i, j] for j in range(len(feature_names))})
        for word, score in custom_scores.items():
            custom_tfidf_matrix[i, feature_names.tolist().index(word)] = score

    svd = TruncatedSVD(n_components=5, )
    data = svd.fit_transform(custom_tfidf_matrix)

    # pca = PCA(n_components=5)
    # reduced_features = pca.fit_transform(tfidf_matrix)
    similarity_matrix = calculate_cosine_similarity(data, docs)

    return similarity_matrix


# Calculate cosine similarity
def calculate_cosine_similarity(matrix, num_docs):
    similarities = [[0 for _ in range(num_docs)] for _ in range(num_docs)]

    for i in range(num_docs):
        for j in range(num_docs):
            similarities[i][j] = cosine_similarity(matrix[i:i + 1], matrix[j:j + 1])[0][0]

    return similarities


def draw(similarities, title):
    draw_heatmap(similarities, title)
    draw_cluster_kmeans(similarities, title)


# Draw heatmaps
def draw_heatmap(similarities, title):
    mask = np.triu(np.ones_like(similarities, dtype=bool))
    np.fill_diagonal(mask, False)

    labels = [repo.split("_")[1] + "/" + repo.split("_")[0] for repo in files_dictionary.keys()]
    plt.figure(figsize=(10, 6))
    sns.heatmap(similarities, annot=True, mask=mask, cmap='flare', xticklabels=labels, yticklabels=labels)
    plt.title('Similarity Between Repository, ' + title)
    plt.xlabel('Repositories')
    plt.ylabel('Repositories')
    # plt.show()
    plt.savefig("../plots/heatmaps/" + title + '4.jpg')


# def draw_cluster(similarities, title):
#     print()

# Draw kmeans clusters
def draw_cluster_kmeans(sim, title):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(sim)

    cluster_labels = kmeans.labels_

    expected_labels = [int(folder) for folder in map((lambda x: x.split("#####")[1]), files_dictionary.keys())]
    # print(expected_labels)

    # print(expected_labels)
    # print(cluster_labels.tolist())
    accuracy_list.append(adjusted_rand_score(expected_labels, cluster_labels.tolist()) * 100)
    # print(accuracy_list)

    # TODO uncomment this for generating graphs
    # pca = PCA(n_components=2)
    # reduced_features = pca.fit_transform(sim)
    #
    # plt.figure(figsize=(8, 6))
    # for i in range(num_clusters):
    #     cluster_points = reduced_features[cluster_labels == i]
    #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
    #
    # for i, txt_file in enumerate(files_dictionary.keys()):
    #     plt.annotate(txt_file, (reduced_features[i, 0], reduced_features[i, 1]))
    # plt.title('Clustering of Repositories, ' + title)
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("../plots/kmeans/" + title + "4.jpg")


def plot_accuracy():
    print(accuracy_list)
    # Plotting the accuracy percentages
    plt.figure(figsize=(8, 6))
    plt.plot(accuracy_list, marker='o', linestyle='-')
    plt.title('Accuracy over different scenarios')
    plt.xlabel('Scenario')
    plt.ylabel('Accuracy (%)')
    scenarios = combinations + ["URLS", "LICENSES"]
    plt.xticks(range(len(scenarios)), scenarios)  # Assigning x-axis ticks
    plt.grid(True)
    # plt.show()
    plt.savefig("../plots/accuracy.jpg")

# Initiate global variables
def initiate_variables():
    input_dir = '../processedDocumentation'
    for root, dirs, files in os.walk(input_dir):
        for dir in dirs:
            files_dictionary[dir] = []
        for file in files:
            if file.endswith('.txt'):
                files_dictionary[root.split("\\")[1]].append(file)


def evaluate_extra_dimension(dim):
    content = [
        read_processed_file("..\\processedDocumentation\\" + folder + "\\" + dim) if dim in files_dictionary[folder]
        else '' for folder in files_dictionary.keys()]
    sim = [[0 for _ in range(len(content))] for _ in range(len(content))]

    for i in range(len(content)):
        for j in range(len(content)):
            if len(set(content[i].split()) | set(content[j].split())) != 0:
                sim[i][j] = len(set(content[i].split()) & set(content[j].split())) / len(
                    set(content[i].split()) | set(content[j].split()))
            else:
                sim[i][j] = 0
    draw_cluster_kmeans(sim, dim.split('.')[0])


# R - readme, W - Wiki, C - comments
combinations = ['R', 'W', 'C', 'RW', 'RC', 'CW', 'RCW']
# combinations = ['W']


def evaluate_similarity(dim):
    if not os.path.exists('../plots'):
        os.makedirs('../plots')
    if not os.path.exists('../plots/heatmaps'):
        os.makedirs('../plots/heatmaps')
    if not os.path.exists('../plots/kmeans'):
        os.makedirs('../plots/kmeans')
    match dim:
        case 'R':
            sim = get_similarity_matrix(['readme.txt'])
            draw(sim, 'Readme only')
        case 'W':
            sim = get_similarity_matrix(['wiki.txt'])
            draw(sim, 'Wiki only')
        case 'C':
            sim = get_similarity_matrix(['comments.txt'])
            draw(sim, 'Comments only')
        case 'RW':
            sim = get_similarity_matrix(['readme.txt', 'wiki.txt'])
            draw(sim, 'Readme + Wiki')
        case 'RC':
            sim = get_similarity_matrix(['readme.txt', 'comments.txt'])
            draw(sim, 'Readme + Comments')
        case 'CW':
            sim = get_similarity_matrix(['comments.txt', 'wiki.txt'])
            draw(sim, 'Comments + Wiki')
        case 'RCW':
            sim = get_similarity_matrix(['readme.txt', 'comments.txt', 'wiki.txt'])
            draw(sim, 'All dimensions')


# TODO  dimensionality reduction - find number of dimensions, cosine distance < 0 ???, add weight to words.  Mostly done
def main():
    initiate_variables()
    print("Initiating variables...")
    for dim in combinations:
        evaluate_similarity(dim)
    evaluate_extra_dimension('urls.txt')
    evaluate_extra_dimension('licenses.txt')
    plot_accuracy()


if __name__ == "__main__":
    main()
