import json
import random
import re
from collections import defaultdict

class NGramModel:
    def __init__(self, n, smoothing_factor=1):
        self.n = n
        self.smoothing_factor = smoothing_factor
        self.ngram_freqs = defaultdict(lambda: defaultdict(int))
        self.context_freqs = defaultdict(int)


    def preprocess_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'[^a-zğüşıöç\s]', '', text)
        return text.split()

    def train(self, data):
     for row in data:
        title_words = self.preprocess_text(row['title'])
        content_words = self.preprocess_text(row['content'])
        words = title_words + content_words
        
        for i in range(len(words) - self.n + 1):
            ngram = tuple(words[i:i + self.n - 1])
            next_word = words[i + self.n - 1]
            
            self.ngram_freqs[ngram][next_word] += 1
            self.context_freqs[ngram] += 1

            if i + self.n - 1 < len(words) - 1:
                next_ngram = tuple(words[i:i + self.n])
                self.context_freqs[next_ngram[:-1]] += 1

    def calculate_probability(self, ngram, next_word):
      ngram_count = self.ngram_freqs[ngram][next_word]
      context_count = self.context_freqs[ngram]
      total_words = len(self.ngram_freqs[ngram])
      if context_count > 0:
        laplace_smoothed_prob = (ngram_count + self.smoothing_factor) / (context_count + total_words * self.smoothing_factor)
        absolute_discount_prob = max(ngram_count - self.smoothing_factor, 0) / context_count
        kneser_ney_prob = (max(ngram_count - self.smoothing_factor, 0) / context_count) + (self.smoothing_factor / context_count) * total_words

        '''smoothed_probability = laplace_smoothed_prob'''
       # smoothed_probability = absolute_discount_prob
        smoothed_probability = kneser_ney_prob
      else:
        smoothed_probability = 1 / total_words

      return smoothed_probability



    def generate_text(self, seed, max_length_factor=8):
        max_length = len(seed) * max_length_factor
        generated_words = seed[:-1]
        for _ in range(max_length):
            current_ngram = tuple(generated_words[-self.n + 1:])
            next_word_candidates = self.ngram_freqs[current_ngram]
            if next_word_candidates:
                next_word_probs = [
                    self.calculate_probability(current_ngram, word)
                    for word in next_word_candidates
                ]
                next_word = random.choices(
                    list(next_word_candidates.keys()),
                    weights=next_word_probs,
                )[0]
                generated_words.append(next_word)
                
                if next_word.endswith('.') or next_word.endswith('!') or next_word.endswith('?'):
                    break  # Sentence boundary detected
            else:
                break
        return ' '.join(generated_words)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():
  while True:
    data_file = 'data.json' 
    n = 2  # Change to the desired n-gram order
    seed = [input('1'), input('2')]  

    data = load_data(data_file)
    model = NGramModel(n)
    model.train(data)

    generated_text = model.generate_text(seed)
    print("nlp:")
    print(generated_text)

if __name__ == '__main__':
    main()
