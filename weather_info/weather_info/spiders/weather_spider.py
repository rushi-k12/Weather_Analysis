import scrapy
from ..items import WeatherInfoItem

class WeatherSpiderSpider(scrapy.Spider):
    name = "weather_spider"
    start_urls = [
        "https://www.timeanddate.com/weather/india"
    ]

    def parse(self, response):
        items = WeatherInfoItem()

        city = response.css('.va-m a::text').extract()
        temperature = response.css('.rbi::text').extract()

        for city, temperature in zip(city,temperature):
            items['city'] = city
            items['temperature'] = temperature

            yield items

