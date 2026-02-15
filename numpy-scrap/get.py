"""
Write a function in Python that
- fetches the given URL,
- parses the HTML content,
- search for each element with class "toctree-l3" and its only child element with tag "a" (a link) and class "reference internal", and copy the href attribute.
- print the list of href attributes.
"""

import requests
from bs4 import BeautifulSoup
import time


def extract_toctree_links(url, toctree_class="toctree-l3"):
    res = requests.get(url)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")
    links = []

    for el in soup.select(f".{toctree_class}"):
        children = el.find_all("a", class_="reference internal", recursive=False)
        if len(children) == 1:
            href = children[0].get("href")
            if href:
                links.append(href)

    return links


# https://numpy.org/doc/stable/reference/routines.html

# Example usage

# # Main namespace
# level = 2
# urs = [
#     "https://numpy.org/doc/stable/reference/constants.html",
#     "https://numpy.org/doc/stable/reference/routines.array-creation.html",
#     "https://numpy.org/doc/stable/reference/routines.array-manipulation.html",
#     "https://numpy.org/doc/stable/reference/routines.bitwise.html",
#     "https://numpy.org/doc/stable/reference/routines.strings.html",
#     "https://numpy.org/doc/stable/reference/routines.datetime.html",
#     "https://numpy.org/doc/stable/reference/routines.dtype.html",
#     "https://numpy.org/doc/stable/reference/routines.emath.html",
#     "https://numpy.org/doc/stable/reference/routines.err.html",
#     "https://numpy.org/doc/stable/reference/routines.exceptions.html",
#     "https://numpy.org/doc/stable/reference/routines.fft.html",
#     "https://numpy.org/doc/stable/reference/routines.functional.html",
#     "https://numpy.org/doc/stable/reference/routines.io.html",
#     "https://numpy.org/doc/stable/reference/routines.indexing.html",
#     "https://numpy.org/doc/stable/reference/routines.linalg.html",
#     "https://numpy.org/doc/stable/reference/routines.logic.html",
#     "https://numpy.org/doc/stable/reference/routines.ma.html",
#     "https://numpy.org/doc/stable/reference/routines.math.html",
#     "https://numpy.org/doc/stable/reference/routines.other.html",
#     "https://numpy.org/doc/stable/reference/routines.polynomials.html",
#     "https://numpy.org/doc/stable/reference/random/index.html",
#     "https://numpy.org/doc/stable/reference/routines.set.html",
#     "https://numpy.org/doc/stable/reference/routines.sort.html",
#     "https://numpy.org/doc/stable/reference/routines.statistics.html",
#     "https://numpy.org/doc/stable/reference/routines.testing.html",
#     "https://numpy.org/doc/stable/reference/routines.window.html",
# ]

# # Random module
# level = 3
# urls = [
#     "https://numpy.org/doc/stable/reference/random/generator.html",
#     "https://numpy.org/doc/stable/reference/random/legacy.html",
#     "https://numpy.org/doc/stable/reference/random/bit_generators/index.html",
#     "https://numpy.org/doc/stable/reference/random/upgrading-pcg64.html",
#     "https://numpy.org/doc/stable/reference/random/compatibility.html",
#     "https://numpy.org/doc/stable/reference/random/parallel.html",
#     "https://numpy.org/doc/stable/reference/random/multithreading.html",
#     "https://numpy.org/doc/stable/reference/random/new-or-different.html",
#     "https://numpy.org/doc/stable/reference/random/performance.html",
#     "https://numpy.org/doc/stable/reference/random/c-api.html",
#     "https://numpy.org/doc/stable/reference/random/extending.html",
# ]


# Polynomials module
level = 3
urls = [
    "https://numpy.org/doc/stable/reference/routines.polynomials.html",
    "https://numpy.org/doc/stable/reference/routines.polynomials.classes.html",
    "https://numpy.org/doc/stable/reference/routines.polynomials.polynomial.html",
    "https://numpy.org/doc/stable/reference/routines.polynomials.chebyshev.html",
    "https://numpy.org/doc/stable/reference/routines.polynomials.hermite.html",
    "https://numpy.org/doc/stable/reference/routines.polynomials.hermite_e.html",
    "https://numpy.org/doc/stable/reference/routines.polynomials.laguerre.html",
    "https://numpy.org/doc/stable/reference/routines.polynomials.legendre.html",
    "https://numpy.org/doc/stable/reference/routines.polynomials.polyutils.html",
    "https://numpy.org/doc/stable/reference/routines.polynomials.poly1d.html",
]


while urls:
    next_urls = []
    for url in urls:
        links = extract_toctree_links(url, f"toctree-l{level+1}")
        print(url)
        for link in links:
            full_link = f"https://numpy.org/doc/stable/reference/{link}"
            print(full_link)
            # next_urls.append(full_link)
        print("", flush=True)
        time.sleep(10)  # Be polite and avoid overwhelming the server
    urls = next_urls
    level += 1
