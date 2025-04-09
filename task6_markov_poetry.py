# *********** Markov Chain Poetry Generator ***********

import random
import string
from collections import defaultdict

# Read the poem
def load_poem(filepath):
    try:
        with open(filepath, 'r') as f:
            text = f.read().lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            lines = [line.split() for line in text.split('\n') if line.strip() != '']
            return lines
    except FileNotFoundError:
        print("Can't find the poem file.")
        return []

# Get how often each word starts a line
def get_start_probs(lines):
    starts = defaultdict(int)
    for line in lines:
        if line:
            starts[line[0]] += 1

    total = len(lines)
    for word in starts:
        starts[word] /= total
    return starts

def first_order_probs(lines):
    probs = defaultdict(lambda: defaultdict(int))
    for line in lines:
        for i in range(len(line) - 1):
            probs[line[i]][line[i+1]] += 1

    for word in probs:
        total = sum(probs[word].values())
        for next_word in probs[word]:
            probs[word][next_word] /= total
    return probs

def second_order_probs(lines):
    probs = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for line in lines:
        for i in range(len(line) - 2):
            w1, w2, w3 = line[i], line[i+1], line[i+2]
            probs[w1][w2][w3] += 1

    for w1 in probs:
        for w2 in probs[w1]:
            total = sum(probs[w1][w2].values())
            for w3 in probs[w1][w2]:
                probs[w1][w2][w3] /= total
    return probs

def pick_word(prob_dict):
    if not prob_dict:
        return "end"
    choices = []
    total = 0
    for word, prob in prob_dict.items():
        total += prob
        choices.append((word, total))

    r = random.random()
    for word, cum_prob in choices:
        if r < cum_prob:
            return word
    return list(prob_dict.keys())[0]

# Create the poem
def make_poem(start_probs, first_probs, second_probs, lines=4):
    poem = []
    for _ in range(lines):
        line = []

        word = pick_word(start_probs)
        line.append(word)

        while len(line) < 10:
            if len(line) == 1:
                word = pick_word(first_probs[line[-1]])
            else:
                w1, w2 = line[-2], line[-1]
                if w1 in second_probs and w2 in second_probs[w1]:
                    word = pick_word(second_probs[w1][w2])
                else:
                    word = pick_word(first_probs[w2])
            line.append(word)

            if word == "end":
                break
        poem.append(' '.join(line))
    return poem

def run():
    lines = load_poem("robert_frost.txt")
    if not lines:
        return

    start_probs = get_start_probs(lines)
    first_probs = first_order_probs(lines)
    second_probs = second_order_probs(lines)

    new_poem = make_poem(start_probs, first_probs, second_probs)
    for line in new_poem:
        print(line)

if __name__ == "__main__":
    run()
