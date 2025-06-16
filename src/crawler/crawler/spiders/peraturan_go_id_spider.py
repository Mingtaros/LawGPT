import scrapy
from bs4 import BeautifulSoup
from furl import furl
import math

class PeraturanSpider(scrapy.Spider):
    name = "peraturangoid"
    allowed_domains = ["peraturan.go.id"]
    start_urls = [
        "https://peraturan.go.id/cari?PeraturanSearch[tentang]=&PeraturanSearch[nomor]=&PeraturanSearch[tahun]=&PeraturanSearch[jenis_peraturan_id]=&PeraturanSearch[pemrakarsa_id]=&PeraturanSearch[status]=Berlaku&page=1",
    ]
            
    
    def parse(self, response):
        # WARNING: this could go a long time
        parsed_html = BeautifulSoup(response.body, 'lxml')
        links = parsed_html.find_all("a", href=True)
        # find number of rules found, find num pages from it
        total_peraturan = parsed_html.find_all("div", class_="col-lg-3 col-md-4 col-10")[0].find_all("strong")[0].text
        total_peraturan = int(total_peraturan.replace(".", ""))
        total_pages = math.ceil(total_peraturan / 20)
        # find number of pages
        for link in links:
            if "/files/" in link["href"]:
                if link["href"] == "/files/draft_kuhap.pdf":
                    continue

                href = link["href"]
                yield {
                    "url": f"https://peraturan.go.id{href}"
                }

        f_url = furl(response.request.url)
        current_page = int(f_url.args["page"])
        if current_page == total_pages:
            # means this is the last page
            return
        else:
            # otherwise, go to next page
            next_page = current_page + 1
            f_url.args["page"] = str(next_page)
            yield scrapy.Request(
                url=f_url.url,
                callback=self.parse
            )
