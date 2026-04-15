import nltk
from nltk import CFG
from nltk.parse import RecursiveDescentParser, ShiftReduceParser
grammar = CFG.fromstring("""
S -> NP VP | S NP
NP -> Det N | N
VP -> V NP | V
Det -> 'the' | 'a'
N -> 'cat' | 'dog' | 'rat'
V -> 'chases' | 'eats'
""")
tokens = ['the', 'cat', 'chases', 'a', 'rat']
rd_parser = RecursiveDescentParser(grammar)
print("\nTop-Down Parsing (Recursive Descent):")
for tree in rd_parser.parse(tokens):
    print(tree)
    tree.pretty_print()
sr_parser = ShiftReduceParser(grammar)
print("\nBottom-Up Parsing (Shift-Reduce):")
for tree in sr_parser.parse(tokens):
    print(tree)
    tree.pretty_print()