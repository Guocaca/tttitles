# tttitles
A neural network that learns to generate interesting book titles.

## Get Training Data
- Crawl: Run `scrapy runspider title_scraper.py -o titles.json` to get book titles from some listopia on Goodreads slike Best Book Titles.
- Clean: Run `python data_preprocess.py titles.json titles.txt` to get rid of the titles with annoying non-ASCII characters and save the clean version as plain text in `titles.txt`.

## Training
- Run `python train_and_save.py` to train a char-RNN model and save it in `*.pt` format.
- Tune the parameters like `learning_rate`, `dropout`, etc.

## Generate some fun book titles
- Run `python sample.py` to read the model from file and generate random fun book titles.