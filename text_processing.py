import re
import os
# import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import requests

url_pattern = re.compile(r'https?://\S+|www\.\S+')

programming_stop_words = ['copyright', 'license', 'licensed', 'param', 'main', 'function', 'static', 'method', 'get',
                          'set', 'todo', 'fixme', 'bug', 'issue', 'pull', 'request', 'merge', 'conflict', 'commit',
                          'push', 'branch', 'master', 'origin', 'remote', 'local', 'repository', 'clone', 'fork', 'git',
                          'github', 'bitbucket', 'gitlab', 'vcs', 'version', 'control', 'system', 'software',
                          'development', 'project', 'team', 'member', 'collaborator', 'contributor', 'author',
                          'reviewer', 'manager', 'leader', 'owner', 'admin', 'administrator', 'class', 'interface',
                          'package']
technical_terms = ['parse', 'print', 'debug', 'exception', 'error', 'warning', 'recognition', 'detection', ]
# spdx_license_identifiers = fetch_spdx_license_list()

stop_words = set(stopwords.words('english'))

stop_words.update(programming_stop_words)


# Extract Licenses

def fetch_spdx_license_list():
    response = requests.get('https://raw.githubusercontent.com/spdx/license-list-data/master/json/licenses.json')
    if response.status_code == 200:
        licenses_data = response.json()
        return [license_data['licenseId'] for license_data in licenses_data['licenses']]
    else:
        return []


# Get the list of SPDX license identifiers

def find_licenses(text):
    text = text.lower()
    licenses = []
    for license_identifier in spdx_license_identifiers:
        if license_identifier.lower() in text:
            licenses.append(license_identifier)
    return set(filter(lambda license: len(license) > 1, licenses))


def isCamelCase(string):
    pattern = r'^[a-zA-Z]+([A-Z][a-z]+)+$'
    return bool(re.match(pattern, string))


def remove_javadoc_tags(string):
    lines = string.split('\n')
    output_lines = []

    for line in lines:
        if line.strip().startswith('@'):
            # Remove the tag part and keep the content after the tag
            tag_index = line.find(' ')
            if tag_index != -1:
                output_lines.append(line[tag_index + 1:])
        else:
            output_lines.append(line)

    return '\n'.join(output_lines)


# TODO: remove auto-generated comments
def process_text(string):
    text_without_url = url_pattern.sub(r'', string)
    removed_javadoc = remove_javadoc_tags(text_without_url)
    removed_license = [token for token in removed_javadoc.split() if token not in spdx_license_identifiers]

    tokens = word_tokenize(' '.join(removed_license))

    # remove punctuation
    cleaned_tokens = [token for token in tokens if token.isalpha()]

    removed_camel_case_tokens = [token for token in cleaned_tokens if not isCamelCase(token)]

    lemmatizer = WordNetLemmatizer()
    lemmatizer_tokens = [lemmatizer.lemmatize(token.lower()) for token in removed_camel_case_tokens]

    filtered_tokens = [token for token in lemmatizer_tokens if (token.lower() not in stop_words)]

    return ' '.join(filtered_tokens)


# Extract URLs

def extract_url(string):
    urls = url_pattern.findall(string)
    filtered_urls = [re.split(r'\s|\[', url)[0] for url in urls]
    cleaned_url = [url.rstrip('\'".,;])') for url in filtered_urls]
    return list(set(cleaned_url))


spdx_license_identifiers = fetch_spdx_license_list()
spdx_license_identifiers = spdx_license_identifiers + [license_token.split("-")[0] for license_token in
                                                       spdx_license_identifiers]

# TODO remove files_dictionary
def main():
    input_dir = 'documentation'
    output_dir = 'processedDocumentation'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files_dictionary = {}
    map_created = False

    for root, dirs, files in os.walk(input_dir):
        if not map_created:
            map_created = True
            for dir in dirs:
                files_dictionary[dir] = []
        urls = []
        licenses = []

        for file in files:
            if file.endswith('.txt'):
                input_file_path = os.path.join(root, file)
                output_file_path = input_file_path.replace(input_dir, output_dir)

                output_folder = os.path.dirname(output_file_path)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                with open(input_file_path, 'r', encoding='utf-8') as input_file:
                    text = input_file.read()
                    processed_text = process_text(text)

                    with open(output_file_path, 'w', encoding='utf-8') as output_file:
                        output_file.write(processed_text)
                urls.append(extract_url(text))
                licenses.append(find_licenses(text))
                files_dictionary[root.split("\\")[1]].append(file)
        if urls:
            output_file_url = os.path.join(root.replace(input_dir, output_dir), 'urls.txt')
            with open(output_file_url, 'w', encoding='utf-8') as output_file:
                for urls_child in urls:
                    for url in urls_child:
                        output_file.write(str(url) + '\n')

            files_dictionary[root.split("\\")[1]].append("url.txt")

        if licenses:
            licenses_set = set()
            for license_child in licenses:
                for license in license_child:
                    licenses_set.add(license)

            output_file_license = os.path.join(root.replace(input_dir, output_dir), 'licenses.txt')
            with open(output_file_license, 'w', encoding='utf-8') as output_file:
                for license in licenses_set:
                    output_file.write(str(license) + '\n')

            files_dictionary[root.split("\\")[1]].append("licenses.txt")


if __name__ == "__main__":
    main()
