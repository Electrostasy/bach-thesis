#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from getopt import getopt
from re import compile
from typing import Set, Any, Iterator
import heapq
import pickle
import sys


def print_inline(object: Any, end="\n") -> None:
    """Print a line of text, then erase it"""
    sys.stdout.write(str(object) + end)
    sys.stdout.write("\x1b[1A")
    sys.stdout.write("\x1b[2K")


def assert_eq_1f(value: float, epsilon: float = 0.0000001) -> None:
    """Assert that value == 1.0, accounting for float inaccuraries"""
    assert value > 1 - epsilon and value < 1 + epsilon


class Token:
    class Category(Enum):
        DIGIT = "D"
        ALPHA = "A"
        SYMBOL = "S"

    _PATTERN = compile(r"[a-zA-Z]+|\d+|\W+")

    def __init__(self, text: str) -> None:
        self.length = len(text)

        if text.isnumeric():
            self.category = Token.Category.DIGIT
        elif text.isalpha():
            self.category = Token.Category.ALPHA
        else:
            self.category = Token.Category.SYMBOL

    def __str__(self) -> str:
        return f"Token({self.category.name},{self.length})"

    def __hash__(self) -> int:
        return hash((self.length, self.category))

    def __eq__(self, other: Token) -> bool:
        length_eq = self.length == other.length
        cat_eq = self.category == other.category
        return length_eq and cat_eq

    @staticmethod
    def tokenize(text: str) -> list[tuple[Token, str]]:
        tokens: list[tuple[Token, str]] = []

        search_results = Token._PATTERN.findall(text.removesuffix("\n"))
        for result in search_results:
            tokens.append((Token(result), result))
        return tokens


class Grammar:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens: list[Token] = tokens

    def __str__(self) -> str:
        g = "".join([f"{t.category.value}{t.length}" for t in self.tokens])
        return f"Grammar({g})"

    def __hash__(self) -> int:
        return hash(tuple(self.tokens))

    def __eq__(self, other: Grammar) -> bool:
        return self.tokens == other.tokens


@dataclass
class QueueItem(tuple):
    grammar: Grammar
    tokens: list[str | Token]
    pivot: int = 0
    frequency: float = 0.0
    _allocations: Set = field(default_factory=set)

    def __le__(self, other) -> bool:
        return self.frequency >= other.frequency

    def __lt__(self, other) -> bool:
        return self.frequency > other.frequency


class PCFG:
    def __init__(self) -> None:
        self.grammars: dict[Grammar, float] = {}
        self.numbers_freqs: dict[Token, tuple[dict[str, float], float]] = {}
        self.symbols_freqs: dict[Token, tuple[dict[str, float], float]] = {}
        self.priority_queue: list[QueueItem] = []

    def pop_preterminal(self) -> list[Token | tuple[str, int]]:
        """
        Pops a preterminal value from the queue, potentially appending new
        preterminal values to the queue based on the popped preterminal
        ("next" function)
        """
        item = heapq.heappop(self.priority_queue)

        pivot = item.pivot
        grammar = item.grammar
        preterm = item.tokens
        frequency = item.frequency

        # Append new derivatives of this value
        for idx, _ in enumerate(preterm[pivot:]):
            match (token := grammar.tokens[idx]).category:
                case Token.Category.ALPHA:
                    # We are not substituting alphas, skip to next token
                    continue
                case Token.Category.DIGIT:
                    # We are substituting this digit, set pivot to idx
                    pivot = idx
                    digits = list(self.digits[token][0].keys())
                    preterm_idx = preterm[pivot][1] + 1
                    if preterm_idx >= len(digits):
                        # Digit counter exceeds length of available digits,
                        # skip this iteration to next token
                        continue
                    preterm[pivot] = digits[preterm_idx], preterm_idx
                case Token.Category.SYMBOL:
                    pivot = idx
                    symbols = list(self.symbols[token][0].keys())
                    preterm_idx = preterm[pivot][1] + 1
                    if preterm_idx >= len(symbols):
                        # Symbol counter exceeds length of available symbols,
                        # skip this iteration to next token
                        continue
                    preterm[pivot] = symbols[preterm_idx], preterm_idx
            heapq.heappush(
                self.priority_queue,
                QueueItem(
                    pivot=pivot,
                    grammar=grammar,
                    tokens=preterm,
                    frequency=frequency / 2
                )
            )

        # for item in self.priority_queue:
        #     sys.stdout.write(f"{item.pivot} => {' '.join([str(t) for t in item.tokens])}\n")

        return item.tokens

    def pass_generator(self, wordlist: list[str]) -> Iterator[str]:
        wordlist_map: dict[Token, set[str]] = process_wordlist(wordlist)

        while len(self.priority_queue) > 0:
            preterminal = self.pop_preterminal()

            # Gather a list of how many alpha tokens are to be substituted
            alphas: list[int] = []
            for idx_t, t in enumerate(preterminal):
                if (isinstance(t, Token) and t.category == Token.Category.ALPHA):
                    alphas.append(idx_t)

            # Initialize them with an initial alpha value
            preterm_copy = preterminal.copy()
            for alpha_idx in alphas:
                words = list(wordlist_map.get(preterm_copy[alpha_idx], []))
                preterminal[alpha_idx] = words[0]

            # Substitute alphas for other alphas
            for alpha_idx in alphas:
                words = wordlist_map.get(preterm_copy[alpha_idx], [])
                for word in words:
                    preterminal[alpha_idx] = word
                    out = ""
                    for p in preterminal:
                        if isinstance(p, tuple):
                            p = p[0]
                        out += str(p)
                    yield out

    def train_with(self, lines: list[str]) -> None:
        """
        Categorizes every symbol/digit/alpha group on every provided line into
        their Token represetation, and initializes a priority queue with
        initial preterminal grammars from these categorized groups
        """
        self.digits: dict[Token, tuple[dict[str, float], float]] = {}
        self.symbols: dict[Token, tuple[dict[str, float], float]] = {}
        for line in lines:
            tokens = list(Token.tokenize(line))
            for token, content in tokens:
                match token.category:
                    case Token.Category.DIGIT:
                        if token not in self.digits:
                            self.digits[token] = ({}, 1.0)
                        value = self.digits[token][0]
                        if content not in value:
                            value[content] = 1.0
                        else:
                            value[content] += 1.0
                        self.digits[token] = value, self.digits[token][1] + 1.0
                    case Token.Category.SYMBOL:
                        if token not in self.symbols:
                            self.symbols[token] = ({}, 1.0)
                        value = self.symbols[token][0]
                        if content not in value:
                            value[content] = 1.0
                        else:
                            value[content] += 1.0
                        self.symbols[token] = value, self.symbols[token][1] + 1.0
            grammar = Grammar([token[0] for token in tokens])
            self.grammars[grammar] = self.grammars.get(grammar, 0.0) + 1.0
        total = sum(self.grammars.values())
        for key in self.grammars.keys():
            self.grammars[key] /= total

        # assert_eq_1f(sum(self.grammars.values()))

        total = sum(value[1] for value in self.digits.values())
        for key in self.digits.keys():
            digits = self.digits[key][0]
            frequency = self.digits[key][1]

            total_digits = sum(digits.values())
            for digit in digits.keys():
                digits[digit] /= total_digits

            # assert_eq_1f(sum(self.digits[key][0].values()))

            self.digits[key] = digits, frequency / total

        # assert_eq_1f(sum(value[1] for value in self.digits.values()))

        total = sum(value[1] for value in self.symbols.values())
        for key in self.symbols.keys():
            symbols = self.symbols[key][0]
            frequency = self.symbols[key][1]

            total_symbols = sum(symbols.values())
            for symbol in symbols.keys():
                symbols[symbol] /= total_symbols

            self.symbols[key] = symbols, frequency / total
        # assert_eq_1f(sum(value[1] for value in self.symbols.values()))

        # Build a priority queue using initial preterminal values
        for grammar, frequency in self.grammars.items():
            preterminal = []
            for token in grammar.tokens:
                match token.category:
                    case Token.Category.DIGIT:
                        terminal = list(self.digits[token][0].keys())[-1]
                        preterminal.append((terminal, 0))
                    case Token.Category.SYMBOL:
                        terminal = list(self.symbols[token][0].keys())[-1]
                        preterminal.append((terminal, 0))
                    case Token.Category.ALPHA:
                        preterminal.append(token)
            heapq.heappush(
                self.priority_queue,
                QueueItem(
                    pivot=0,
                    grammar=grammar,
                    tokens=preterminal,
                    frequency=frequency
                )
            )


def process_wordlist(lines: list[str]) -> dict[Token, set[str]]:
    """
    Categorizes each word on every line without symbols or digits and returns a
    mapping of their Token representation to list of words
    """
    words = {}
    line_tokens = [Token.tokenize(line) for line in lines]
    for line in line_tokens:
        for token, str in line:
            if token.category == Token.Category.ALPHA:
                words[token] = words.get(token, []) + [str]
    return words


def print_help(exit_code: int) -> None:
    print("-h: show this help")
    print("-f: load PCFG from file")
    print("-o: store PCFG to file")
    print("-l: limit generated passwords to number")
    print("-s: export generated passwords (by default stdout)")
    print("-t: password file for training")
    print("-w: password/wordlist for alpha strings")
    print("-r: limit training to passwords of length a,b")
    exit(exit_code)


if __name__ == '__main__':
    '''
    Train a model and export it:
        $ ./pcfg.py -t training.txt -r 6,12 -o model.pickle
    Load a model and generate guesses:
        $ ./pcfg.py -l 10000 -w wordlist.txt -f model.pickle -s guesses_10000.txt

    Combine wordlists (password dicts) with:
        $ cat wordlist1.txt wordlist2.txt | shuf > combined.txt
    The wordlist processor can process passwords into alphas tokens as well:
        $ ./split.fish -i rockyou.txt -t 0.8 -w 0.2
        $ # creates `training.txt` and `wordlist.txt` accordingly
    '''
    limit = 1_000
    out_file = ''
    train_file = ''
    words_file = ''
    password_range = 6, 12

    args = sys.argv[1:]
    if len(args) == 0:
        print_help(1)

    pcfg = PCFG()
    serialize_to: str | None = None
    deserialize_from: str | None = None

    opts, _ = getopt(args, shortopts='hf:o:l:s:t:w:r:')
    for option, value in opts:
        match option:
            case '-h':
                print_help(0)
            case '-f':
                deserialize_from = value
            case '-o':
                serialize_to = value
            case '-l':
                limit = int(value)
            case '-s':
                out_file = value
            case '-t':
                train_file = value
            case '-w':
                words_file = value
            case '-r':
                password_range = tuple([int(x) for x in value.split(',')])
                assert len(password_range) == 2

    if serialize_to is not None:
        if train_file == '':
            print('Provide a training file with `-t`')
            exit(1)

        with open(train_file, 'r', encoding='utf8') as f:
            lines = [line.removesuffix('\n') for line in f.readlines() if len(line) in range(*password_range)]
            pcfg.train_with(lines)

        with open(serialize_to, 'wb') as dest_file:
            pickle.dump(pcfg, dest_file)
        exit(0)

    if deserialize_from is not None:
        with open(deserialize_from, 'rb') as source_file:
            pcfg = pickle.load(source_file)

    if words_file == '':
        print('Provide a wordlist file with `-w`')
        exit(1)

    wordlist = []
    with open(words_file, 'r', encoding='utf8') as f:
        wordlist = list(set(f.readlines()))

    with open(out_file, 'w') as f:
        for i, password in enumerate(pcfg.pass_generator(wordlist)):
            if i > limit:
                break
            print_inline(f"Generated password: {password}, {len(pcfg.priority_queue)} terminals in queue")
            f.write(password)
            f.write('\n')
