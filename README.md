# tttitles
A neural network that learns to generate interesting book titles.

## Get Training Data
- Crawl: Run `scrapy runspider title_scraper.py -o titles.json` to get book titles from some listopia on Goodreads with like Best Book Titles.
- Clean: Run `python data_preprocess.py titles.json titles.txt` to get rid of annoying non-ASCII characters in some of the titles and save the clean version as plain text in `titles.txt`.

## Training

## Generate some fun book titles