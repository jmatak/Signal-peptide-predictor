class Sequence:
    def __init__(self, input, output=None, label=None, name=None):
        self.name = name
        self.input = input
        self.output = output
        self.label = label

    def __repr__(self):
        return print_fasta(self.name, self.input)


def break_sequence(sequence, n):
    return [sequence[i:i + n] for i in range(0, len(sequence), n)]


def print_fasta(name, sequence, FASTA_LENGTH=60):
    sequence = break_sequence(sequence, FASTA_LENGTH)
    s = [">" + name]
    s.extend(sequence)
    return "\n".join(s)
