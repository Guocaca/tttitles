import sys
import json
import string

titles_json = sys.argv[1]
titles_text = sys.argv[2]
key = sys.argv[3]
printable = set(string.printable)

with open(titles_json) as json_file:
    data = json.load(json_file)
json_file.close()

titles_set = set()

text_file = open(titles_text, 'w')

for item in data:
    original = item[key]
    # Specifically added this for mobile game names
    original = original.replace('\u2122', '').replace('\u00ae', '').replace('\u2013', '')
    if not any(map(lambda x: x not in printable, original)):
        lower = original.lower()
        if lower not in titles_set:
            titles_set.add(lower)
            text_file.write(lower + '\n')
text_file.close()
