class Tokenizer():
    """
    Utility class for model tokenisation
    """
    vocab: str
    stoi: dict[str, int]
    itos: dict[int, str]

    def __init__(self, text: str) -> str:
        """ 
        :param text: Vocab gets generated given this text (likely the dataset)
        """
        self.vocab = ''.join(sorted(list(set(text))))
        self.stoi = { ch:i for i,ch in enumerate(self.vocab) }
        self.itos = { i:ch for i,ch in enumerate(self.vocab) }
        
    def encode(self, text: str) -> list[int]:
        """
        Generates tokens out of the vocab given text
        """
        return [self.stoi[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        """
        Decodes the tokens to human readable string out of the vocab
        """
        return ''.join([self.itos[i] for i in tokens])

