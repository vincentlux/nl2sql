def count_lines(fname):
    with open(fname) as f:
        print(sum(1 for line in f))
        return sum(1 for line in f)

