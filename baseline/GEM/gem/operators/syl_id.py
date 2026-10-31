import random


class SyllableIDGenerator:
    def __init__(self, syllables=None, num_parts=2, separator=''):
        self.syllables = syllables or [
            "ba", "be", "bi", "bo", "bu",
            "ka", "ke", "ki", "ko", "ku",
            "la", "le", "li", "lo", "lu",
            "ra", "re", "ri", "ro", "ru",
            "za", "ze", "zi", "zo", "zu",
            "an", "en", "in", "on", "un",
            "mar", "tor", "zan", "vor", "xel", "grim", "dra"
        ]
        self.num_parts = num_parts
        self.separator = separator

    def generate(self):
        parts = [random.choice(self.syllables) for _ in range(self.num_parts)]
        return self.separator.join(parts)


# Example usage
if __name__ == "__main__":
    gen = SyllableIDGenerator(num_parts=4)
    print(gen.generate())
