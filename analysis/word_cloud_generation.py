import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

def generate_word_cloud(text, file_name):
    mask = np.array(Image.open("circle.png"))

    # Create and generate a word cloud image:
    wc = WordCloud(stopwords=set(['?',',']), collocations=True, max_font_size=1000, max_words=100, background_color="white").generate(text)
    # wc = WordCloud( collocations=False, max_font_size=1000, max_words=100, background_color="white", mask=mask).generate(text)
    wc.to_file('{0}.png'.format(file_name))

    # Display the generated image:
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def call_toy_example():
# https://www.datacamp.com/community/tutorials/wordcloud-python
    df = pd.read_csv("toy.csv", index_col=0)

    # Groupby by country
    country = df.groupby("country")
    print country.describe().head()

    # plt.figure(figsize=(15,10))
    # country.size().sort_values(ascending=False).plot.bar()
    # plt.xticks(rotation=50)
    # plt.xlabel("Country of Origin")
    # plt.ylabel("Number of Wines")
    # plt.show()

    # Start with one review:
    text = 'With over 81 million inhabitants Iran is the world 18th most populous country Iran also called Persia and officially known as the Islamic Republic of Iran  is a country in Western Asia' \
           'Canada climate varies widely across its vast area Canada is a country in the northern part of North America With over 81 million inhabitants Iran is the world 18th most populous country'


    mask = np.array(Image.open("circle.png"))

    # Create and generate a word cloud image:
    wc = WordCloud(max_font_size=1000, max_words=100, background_color="white", mask=mask).generate(text)
    wc.to_file("history.png")

    # Display the generated image:
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    call_toy_example()