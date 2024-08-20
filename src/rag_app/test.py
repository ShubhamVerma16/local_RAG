import ollama
import os

os.getenv('OLLAMA_HOST')
print("================")
print(os.getenv('OLLAMA_HOST'))

import time
now = time.time()

response = ollama.chat(model='llama3.1', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
later = time.time()
difference = int(later - now)
print(difference)
# print(response['message']['content'])