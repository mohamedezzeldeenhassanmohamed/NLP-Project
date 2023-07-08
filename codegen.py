#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import warnings
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

# Set of visited URLs to avoid duplicate scraping
visited_urls = set()

# Set of seen document content to avoid repeating output
seen_doc_content = set()

def scrape_page(url, max_depth=1, current_depth=0):
    # Get the HTML content of the URL
    response = requests.get(url)
    html_content = response.content

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the main content section of the page
    main_content = soup.find('div', {'class': 'section'})

    if main_content is None:
        # No main content section found, skip scraping this page
        return

    # Extract the relevant content from the main content section
    doc_content = '\n'.join(element.text.strip() for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'p']))

    # Display the extracted content to the user
    if doc_content not in seen_doc_content:
        print(doc_content)
        seen_doc_content.add(doc_content)

    # Find all the links within the page
    urls = [urljoin(url, link.get('href')) for link in soup.find_all('a') if link.get('href') and link.parent.name != 'div' and link.parent.get('class') != 'wy-nav-content' and link['href'].startswith(url)]

    # Recursively scrape each link
    with ThreadPoolExecutor() as executor:
        futures = []
        for link in urls:
            if current_depth < max_depth and link not in visited_urls:
                visited_urls.add(link)
                future = executor.submit(scrape_page, link, max_depth, current_depth + 1)
                futures.append(future)

        # Wait for all tasks to complete
        for future in futures:
            future.result()

# Get the initial URL
initial_url = input("Enter the URL: ")

# Start scraping from the initial URL
scrape_page(initial_url, max_depth=3)

# Open the text file for writing
with open('output.txt', 'w') as file:
    file.write('\n'.join(seen_doc_content))


# In[4]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

with open("output.txt", "r") as file:
    text = file.read()

inputs = tokenizer([text], truncation=True, padding=True, max_length=1024, return_tensors="pt")
summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=100, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)

with open("summ.txt", "w") as file:
    file.write(summary)


# In[5]:


from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small-code-to-text")
model = AutoModelForCausalLM.from_pretrained("codeparrot/codeparrot-small-code-to-text")

def gencode(txt):
    
    # Tokenize the prompt
    input_ids = tokenizer.encode(txt, return_tensors="pt")

    # Generate code
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, num_beams=5)

    # Decode the generated code
    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)

    # Print the generated code
    return generated_code


# In[ ]:





# In[ ]:




