import os


# Get the top 10 most similar pairs for each scenario
def get_top_sim(input_file_path):
    content = []
    print(input_file_path)
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            elements = line.strip().split(' ')
            tuple_l = (elements[0], elements[1], float(elements[2]))

            content.append(tuple_l)
    filtered_content = [el for el in content if el[0] != el[1]]
    sorted_content = sorted(filtered_content, key=lambda x: x[2], reverse=True)

    returned = []
    index = 0
    while len(returned) < 10:
        curr = sorted_content[index]
        inverse = (curr[1], curr[0], curr[2])
        if curr not in returned and inverse not in returned:
            returned.append(curr)
        index += 1
    return returned


def main():
    final = []
    titles = []
    for root, dirs, files in os.walk('../plots/processedCrossSim'):
        for file in files:
            if file != 'CrossSimEval.txt':
                titles.append(file.strip().split('.')[0])
                final.append(get_top_sim(os.path.join(root, file)))
    with open('../plots/processedCrossSim' + '/CrossSimEval.txt', 'w') as file:
        for index, curr in enumerate(final):
            file.write('%s\n' % titles[index])
            for line in curr:
                file.write('%s\n' % str(line))


if __name__ == "__main__":
    main()
