import sys

# Get all the possible letters and all training samples in a list
def readFile(file_name):
    all_letters = set()
    all_titles = open(file_name).read().strip().split('\n')
    for title in all_titles:
        all_letters = all_letters.union(set(title))
    return ''.join(list(all_letters)), all_titles

file_name = sys.argv[1]
all_letters, all_titles = readFile(file_name)
n_letters = len(all_letters) + 1 # For EOS