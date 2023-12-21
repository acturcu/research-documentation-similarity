import os
import numpy as np
import matplotlib.pyplot as plt


def count_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        tokens = content.split(" ")
        return len(tokens)


def calculate_percentage(count, total):
    return (count / total) * 100


def main():
    folders = os.listdir("processedDocumentation")

    readme_percentages = []
    comments_percentages = []
    wiki_percentages = []
    for folder in folders:
        readme_count = 0
        comments_count = 0
        wiki_count = 0

        folder_path = os.path.join("processedDocumentation", folder)
        files = os.listdir(folder_path)
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(folder_path, file)
                if file == 'readme.txt':
                    readme_count += count_words(file_path)
                elif file == 'comments.txt':
                    comments_count += count_words(file_path)
                elif file == 'wiki.txt':
                    wiki_count += count_words(file_path)

        total_count = readme_count + comments_count + wiki_count

        readme_percentage = calculate_percentage(readme_count, total_count)
        comments_percentage = calculate_percentage(comments_count, total_count)
        wiki_percentage = calculate_percentage(wiki_count, total_count)

        readme_percentages.append(readme_percentage)
        comments_percentages.append(comments_percentage)
        wiki_percentages.append(wiki_percentage)

    counts = {
        "Readme": readme_percentages,
        "Comments": comments_percentages,
        "Wiki": wiki_percentages
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(folders))

    for doc, count in counts.items():
        p = ax.bar(folders, count, bottom=bottom, label=doc)
        bottom += count

        ax.bar_label(p, label_type='center')

    ax.set_title('Percentage of Each Documentation Type')
    ax.legend()
    plt.xticks(rotation=90)
    plt.show()


if __name__ == '__main__':
    main()
