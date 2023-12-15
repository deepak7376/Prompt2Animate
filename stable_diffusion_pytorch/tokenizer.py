import re

class SimpleTokenizer:
    def __init__(self):
        self.pattern = re.compile(r'\w+')

    def tokenize(self, text):
        return self.pattern.findall(text.lower())

if __name__ == "__main__":
    # Example usage
    tokenizer = SimpleTokenizer()
    text = "This is a simple example sentence. Tokenize me!"

    tokens = tokenizer.tokenize(text)
    print(tokens)
