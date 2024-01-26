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
from datetime import datetime
import random

files_dictionary = {}
repo_clusters = {}
num_clusters = 5
licenses = []
urls = []

# terms that are more likely to be used in technical documentation, and thus should be given more weight
technical_terms = ['parse', 'print', 'debug', 'exception', 'error', 'warning', 'recognition', 'detection', 'test',
                   'mod', 'game', 'render', ]
accuracy_list = []


def read_processed_file(location):
    content = []
    with open(location, 'r', encoding='utf-8') as file:
        content.append(file.read())
    return ' '.join(content)


# urls and licenses are not passed anymore
def read_processed_folder(directory, files, input_dir):
    content = []
    for file in files:
        if file in files_dictionary[directory]:
            location = os.path.join("../" + input_dir, directory, file)
            content.append(read_processed_file(location))
        else:
            content.append('')
    return ' '.join(content)


# Increase the weight of selected words in the TF-IDF vector
def modify_weights(doc):
    return {word: tfidf * 1.5 if word in technical_terms else tfidf for word, tfidf in doc.items()}


# Calculate the similarity matrix using TF-IDF
def get_similarity_matrix(files, input_dir):
    content = [read_processed_folder(folder, files, input_dir) for folder in files_dictionary.keys()]
    docs = len(content)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(content)
    print('TFIDF performed')
    feature_names = tfidf_vectorizer.get_feature_names_out()
    custom_tfidf_matrix = sp.lil_matrix(tfidf_matrix.shape, dtype=tfidf_matrix.dtype)
    for i in range(docs):
        custom_scores = modify_weights({feature_names[j]: tfidf_matrix[i, j] for j in range(len(feature_names))})
        for word, score in custom_scores.items():
            custom_tfidf_matrix[i, feature_names.tolist().index(word)] = score
    #
    svd = TruncatedSVD(n_components=5, )
    data = svd.fit_transform(custom_tfidf_matrix)
    print('dimensionality reduction performed')
    similarity_matrix = calculate_cosine_similarity(data, docs)

    # plot_accuracy_three_options(data, docs)
    print('similarity matrix calculated')
    return similarity_matrix


# Calculate cosine similarity
def calculate_cosine_similarity(matrix, num_docs):
    similarities = [[0.5 for _ in range(num_docs)] for _ in range(num_docs)]

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
    # plt.figure(figsize=(10, 6))
    sns.heatmap(similarities, annot=True, mask=mask, cmap='flare', xticklabels=labels, yticklabels=labels)
    plt.title('Similarity Between Repository, ' + title)
    plt.xlabel('Repositories')
    plt.ylabel('Repositories')
    plt.savefig("../plots/heatmaps/" + title + '.jpg')
    plt.close()


# Draw kmeans clusters
def draw_cluster_kmeans(sim, title):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(sim)

    cluster_labels = kmeans.labels_

    expected_labels = [int(folder) for folder in map((lambda x: x.split("#####")[1]), files_dictionary.keys())]

    print(cluster_labels.tolist())
    accuracy_list.append(adjusted_rand_score(expected_labels, cluster_labels.tolist()) * 100)

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(sim)

    plt.figure(figsize=(8, 6))
    for i in range(num_clusters):
        cluster_points = reduced_features[cluster_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

    for i, txt_file in enumerate(files_dictionary.keys()):
        plt.annotate(txt_file, (reduced_features[i, 0], reduced_features[i, 1]))
    plt.title('Clustering of Repositories, ' + title)
    plt.xlabel('Similarity distance')
    plt.ylabel('Similarity distance')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig("../plots/kmeans/" + title + ".jpg")
    plt.close()


# Plot the accuracy percentages of kmeans clustering
def plot_accuracy():
    print(accuracy_list)
    plt.figure(figsize=(8, 6))
    # plt.plot(accuracy_list, marker='o', linestyle='-')
    plt.bar(range(len(accuracy_list)), accuracy_list)
    plt.title('K Means Clustering')
    plt.xlabel('Scenario')
    plt.ylabel('Accuracy (%)')
    scenarios = combinations + ["URLS", "LICENSES"]
    plt.xticks(range(len(scenarios)), scenarios)
    plt.grid(True)
    plt.savefig("../plots/accuracy.jpg")
    plt.close()

    # plt.figure(figsize=(12, 9))
    # plt.rcParams.update({'font.size': 12})
    # bar_positions1 = np.arange(len(accuracy_list0))
    # bar_positions2 = bar_positions1 + 0.25
    # bar_positions3 = bar_positions1 + 0.5
    # bar_width = 0.25
    #
    # plt.bar(bar_positions1, accuracy_list0, width=bar_width, label='Initial similarity score 0')
    # plt.bar(bar_positions2, accuracy_list05, width=bar_width, label='Initial simialrit score 0.5')
    # plt.bar(bar_positions3, accuracy_list1, width=bar_width, label='Initial similarity score 1')
    #
    #
    #
    # plt.title('K Means Clustering Accuracy')
    # plt.xlabel('Scenario')
    # plt.ylabel('Accuracy (%)')
    # scenarios = ['Readme only', 'Wiki only', 'Comments only', 'Readme + Wiki', 'Readme + Comments', 'Comments + Wiki', 'All dimensions'] + ["URLS", "LICENSES"]
    # plt.xticks(bar_positions1 + bar_width, scenarios, rotation=15)
    # plt.grid(True)
    # plt.legend(loc='best')
    # # plt.show()
    # current_time = datetime.now().time()
    # mta = random.randint(0, 100000000)
    # plt.savefig("../plots/accuracyAll" + str(current_time).split(':')[1] + str(mta) + ".jpg")
    # plt.close()


# accuracy_list0 = []
# accuracy_list05 = []
# accuracy_list1 = []
# def plot_accuracy_three_options(matrix, num_docs):
#
#     similarities0 = [[0 for _ in range(num_docs)] for _ in range(num_docs)]
#     for i in range(num_docs):
#         for j in range(num_docs):
#             similarities0[i][j] = cosine_similarity(matrix[i:i + 1], matrix[j:j + 1])[0][0]
#     similarities05 = [[0.5 for _ in range(num_docs)] for _ in range(num_docs)]
#     for i in range(num_docs):
#         for j in range(num_docs):
#             similarities05[i][j] = cosine_similarity(matrix[i:i + 1], matrix[j:j + 1])[0][0]
#
#     similarities1 = [[1 for _ in range(num_docs)] for _ in range(num_docs)]
#     for i in range(num_docs):
#         for j in range(num_docs):
#             similarities1[i][j] = cosine_similarity(matrix[i:i + 1], matrix[j:j + 1])[0][0]
#
#     expected_labels = [int(folder) for folder in map((lambda x: x.split("#####")[1]), files_dictionary.keys())]
#
#     kmeans0 = KMeans(n_clusters=num_clusters)
#     kmeans0.fit(similarities0)
#
#     cluster_labels0 = kmeans0.labels_
#
#     accuracy_list0.append(adjusted_rand_score(expected_labels, cluster_labels0.tolist()) * 100)
#
#     kmeans05 = KMeans(n_clusters=num_clusters)
#     kmeans05.fit(similarities05)
#
#     cluster_labels05 = kmeans05.labels_
#
#     accuracy_list05.append(adjusted_rand_score(expected_labels, cluster_labels05.tolist()) * 100)
#
#     kmeans1 = KMeans(n_clusters=num_clusters)
#     kmeans1.fit(similarities1)
#
#     cluster_labels1 = kmeans1.labels_
#
#     accuracy_list1.append(adjusted_rand_score(expected_labels, cluster_labels1.tolist()) * 100)


# Initiate global variables
def initiate_variables(input_dir):
    files_dictionary.clear()
    input_dir = '../' + input_dir
    for root, dirs, files in os.walk(input_dir):
        for dir in dirs:
            files_dictionary[dir] = []
        for file in files:
            if file.endswith('.txt'):
                files_dictionary[root.split("\\")[1]].append(file)


def evaluate_extra_dimension(dim, directory):
    print('extra dimension ' + dim)
    content = [
        read_processed_file("..\\" + directory + "\\" + folder + "\\" + dim) if dim in files_dictionary[folder]
        else '' for folder in files_dictionary.keys()]
    sim = [[0.5 for _ in range(len(content))] for _ in range(len(content))]

    for i in range(len(content)):
        for j in range(len(content)):
            if len(set(content[i].split()) | set(content[j].split())) != 0:
                sim[i][j] = len(set(content[i].split()) & set(content[j].split())) / len(
                    set(content[i].split()) | set(content[j].split()))
            else:
                sim[i][j] = 0.5
    draw_cluster_kmeans(sim, dim.split('.')[0])
    # sim0 = [[0 for _ in range(len(content))] for _ in range(len(content))]
    #
    # for i in range(len(content)):
    #     for j in range(len(content)):
    #         if len(set(content[i].split()) | set(content[j].split())) != 0:
    #             sim0[i][j] = len(set(content[i].split()) & set(content[j].split())) / len(
    #                 set(content[i].split()) | set(content[j].split()))
    #         else:
    #             sim0[i][j] = 0
    #

    # sim05 = [[0.5 for _ in range(len(content))] for _ in range(len(content))]
    #
    # for i in range(len(content)):
    #     for j in range(len(content)):
    #         if len(set(content[i].split()) | set(content[j].split())) != 0:
    #             sim05[i][j] = len(set(content[i].split()) & set(content[j].split())) / len(
    #                 set(content[i].split()) | set(content[j].split()))
    #         else:
    #             sim05[i][j] = 0.5
    #
    # sim1 = [[1 for _ in range(len(content))] for _ in range(len(content))]
    #
    # for i in range(len(content)):
    #     for j in range(len(content)):
    #         if len(set(content[i].split()) | set(content[j].split())) != 0:
    #             sim1[i][j] = len(set(content[i].split()) & set(content[j].split())) / len(
    #                 set(content[i].split()) | set(content[j].split()))
    #         else:
    #             sim1[i][j] = 1
    #
    # kmeans = KMeans(n_clusters=num_clusters)
    # kmeans.fit(sim0)
    # cluster_labels = kmeans.labels_
    # expected_labels = [int(folder) for folder in map((lambda x: x.split("#####")[1]), files_dictionary.keys())]
    # accuracy_list0.append(adjusted_rand_score(expected_labels, cluster_labels.tolist()) * 100)
    #
    # kmeans5 = KMeans(n_clusters=num_clusters)
    # kmeans5.fit(sim05)
    # cluster_labels5 = kmeans5.labels_
    # accuracy_list05.append(adjusted_rand_score(expected_labels, cluster_labels5.tolist()) * 100)
    #
    # kmeans1 = KMeans(n_clusters=num_clusters)
    # kmeans1.fit(sim1)
    # cluster_labels1 = kmeans1.labels_
    # accuracy_list1.append(adjusted_rand_score(expected_labels, cluster_labels1.tolist()) * 100)


# save the similarity matrix in a file, a heatmap was too dense to read
def pretty_print_matrix(sim_matrix, title, directory):
    labels = [repo.split("_")[1] + "/" + repo.split("_")[0] for repo in files_dictionary.keys()]
    num_docs = len(sim_matrix)

    content = []
    for i in range(num_docs):
        for j in range(num_docs):
            content.append(labels[i] + ' ' + labels[j] + ' ' + str(sim_matrix[i][j]))

    destination = '../plots/' + directory

    if not os.path.exists(destination):
        os.makedirs(destination)

    with open(destination + '/' + title + '.txt', 'w') as file:
        for line in content:
            file.write('%s\n' % line)


# R - readme, W - Wiki, C - comments
combinations = ['R', 'W', 'C', 'RW', 'RC', 'CW', 'RCW']

# Evaluate the similarity of repositories for all combinations of dimensions
def evaluate_similarity(dim, directory):
    if not os.path.exists('../plots'):
        os.makedirs('../plots')
    if not os.path.exists('../plots/heatmaps'):
        os.makedirs('../plots/heatmaps')
    if not os.path.exists('../plots/kmeans'):
        os.makedirs('../plots/kmeans')
    experiment_type = directory == 'processedDocumentation'
    match dim:
        case 'R':
            sim = get_similarity_matrix(['readme.txt'], directory)
            if experiment_type:
                draw(sim, 'Readme only')
            pretty_print_matrix(sim, 'Readme Only', directory)
        case 'W':
            sim = get_similarity_matrix(['wiki.txt'], directory)
            if experiment_type:
                draw(sim, 'Wiki only')
            pretty_print_matrix(sim, 'Wiki Only', directory)
        case 'C':
            sim = get_similarity_matrix(['comments.txt'], directory)
            if experiment_type:
                draw(sim, 'Comments only')
            pretty_print_matrix(sim, 'Comments Only', directory)
        case 'RW':
            sim = get_similarity_matrix(['readme.txt', 'wiki.txt'], directory)
            if experiment_type:
                draw(sim, 'Readme + Wiki')
            pretty_print_matrix(sim, 'Readme + Wiki', directory)
        case 'RC':
            sim = get_similarity_matrix(['readme.txt', 'comments.txt'], directory)
            if experiment_type:
                draw(sim, 'Readme + Comments')
            pretty_print_matrix(sim, 'Readme + Comments', directory)
        case 'CW':
            sim = get_similarity_matrix(['comments.txt', 'wiki.txt'], directory)
            if experiment_type:
                draw(sim, 'Comments + Wiki')
            pretty_print_matrix(sim, 'Comments + Wiki', directory)
        case 'RCW':
            sim = get_similarity_matrix(['readme.txt', 'comments.txt', 'wiki.txt'], directory)
            if experiment_type:
                draw(sim, 'All dimensions')
            pretty_print_matrix(sim, 'All dimensions', directory)


# call all functions related to the directory
def calculate_similarity(directory):
    initiate_variables(directory)
    print("Initiating variables...")
    for dim in combinations:
        evaluate_similarity(dim, directory)
    if directory == 'processedDocumentation':
        evaluate_extra_dimension('urls.txt', directory)
        evaluate_extra_dimension('licenses.txt', directory)

        plot_accuracy()


def main():
    calculate_similarity('processedDocumentation')
    calculate_similarity('processedCrossSim')


if __name__ == "__main__":
    main()
