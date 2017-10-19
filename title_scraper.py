import scrapy

listopia = ["https://www.goodreads.com/list/show/276.Best_Book_Titles?page=" + str(i) for i in range(1, 27)]
listopia += ["https://www.goodreads.com/list/show/5775.I_Picked_It_Up_Because_of_the_Title?page=" + str(i) for i in range(1, 12)]
listopia += ["https://www.goodreads.com/list/show/23000.Best_Self_Published_Short_Story_Titles?page=" + str(i) for i in range(1, 4)]
listopia += ["https://www.goodreads.com/list/show/21786.Eye_Catching_Titles?page=" + str(i) for i in range(1, 4)]
listopia += ["https://www.goodreads.com/list/show/4298.There_Ought_to_be_a_Band?page=" + str(i) for i in range(1, 22)]
listopia += ["https://www.goodreads.com/list/show/706.Most_Poetic_Book_Titles?page=" + str(i) for i in range(1, 13)]

class BookTitleSpider(scrapy.Spider):
    name = "book_title_spider"
    start_urls = listopia

    def parse(self, response):
        item_selector = '//tr[@itemscope]'
        title_selector = 'a ::attr(title)'
        for book in response.xpath(item_selector):
            yield {
                "title": book.css(title_selector).extract_first(),
            }