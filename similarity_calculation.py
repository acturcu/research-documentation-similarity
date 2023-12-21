import os

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
import seaborn as sns
import numpy as np

# pip install scikit-learn
# pip install nltk
# pip install requests
#  pip install matplotlib
# pip install seaborn

files_dictionary = {}
licenses = []
urls = []


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
            location = os.path.join("processedDocumentation", directory, file)
            content.append(read_processed_file(location))
        else:
            content.append('')
    return ' '.join(content)


def get_similarity_matrix(files):
    content = [read_processed_folder(folder, files) for folder in files_dictionary.keys()]

    docs = len(content)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(content)
    svd = TruncatedSVD(n_components=5, )
    data = svd.fit_transform(tfidf_matrix)

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
    # draw_cluster(similarities, title)
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
    plt.savefig("plots/heatmaps/" + title + '2.jpg')


# def draw_cluster(similarities, title):
#     print()

# Draw kmeans clusters
def draw_cluster_kmeans(sim, title):
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(sim)

    cluster_labels = kmeans.labels_

    # files_with_clusters = {file: label for file, label in zip(files_dictionary.keys(), cluster_labels)}
    # print(files_with_clusters)

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(sim)

    plt.figure(figsize=(8, 6))
    for i in range(num_clusters):
        cluster_points = reduced_features[cluster_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

    for i, txt_file in enumerate(files_dictionary.keys()):
        plt.annotate(txt_file, (reduced_features[i, 0], reduced_features[i, 1]))
    plt.title('Clustering of Repositories, ' + title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/kmeans/" + title + "2.jpg")

    # TODO 3D
    # pca = PCA(n_components=3)
    # reduced_features = pca.fit_transform(sim)
    #
    #
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # for i in range(num_clusters):
    #     cluster_points = reduced_features[cluster_labels == i]
    #     ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i}')
    #
    #
    # for i, txt_file in enumerate(files_dictionary.keys()):
    #     ax.text(reduced_features[i, 0], reduced_features[i, 1], reduced_features[i, 2], txt_file)
    #
    # ax.set_title('Clustering of Repositories in 3D. ' + title)
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # ax.set_zlabel('PC3')
    # ax.legend()
    # plt.savefig("plots/kmeans/" + title + ".jpg")


# Initiate global variables
def initiate_variables():
    input_dir = 'processedDocumentation'
    for root, dirs, files in os.walk(input_dir):
        for dir in dirs:
            files_dictionary[dir] = []
        for file in files:
            if file.endswith('.txt'):
                files_dictionary[root.split("\\")[1]].append(file)
    # #     files_dictionary[key] = read_processed_folder(key, files_dictionary[key])
    print(files_dictionary.keys())

def evaluate_extra_dimension(dim):
    content = [read_processed_file("processedDocumentation\\" + folder + "\\" + dim) if dim in files_dictionary[folder]
               else '' for folder in files_dictionary.keys()]
    sim =[[0 for _ in range(len(content))] for _ in range(len(content))]

    for i in range(len(content)):
        for j in range(len(content)):
            sim[i][j] = len(set(content[i].split()) & set(content[j].split())) / len(set(content[i].split()) | set(content[j].split()))
    draw_cluster_kmeans(sim, dim.split('.')[0])


# R - readme, W - Wiki, C - comments
combinations = ['R', 'W', 'C', 'RW', 'RC', 'CW', 'RCW']


def evaluate_similarity(dim):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    if not os.path.exists('plots/heatmaps'):
        os.makedirs('plots/heatmaps')
    if not os.path.exists('plots/kmeans'):
        os.makedirs('plots/kmeans')
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


# TODO  dimensionality reduction - find number of dimensions, cosine distance < 0 ???, add weight to words
def main():
    initiate_variables()
    print("Initiating variables...")
    for dim in combinations:
        evaluate_similarity(dim)
    evaluate_extra_dimension('urls.txt')
    evaluate_extra_dimension('licenses.txt')

if __name__ == "__main__":
    main()
