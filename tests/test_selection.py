from qarchga.selection import Individual, elitism

def test_elitism():
    pop = [
        Individual(genome=None, fitness=1.0, depth=1, n2q=0, age=0),
        Individual(genome=None, fitness=2.0, depth=1, n2q=0, age=0),
        Individual(genome=None, fitness=0.5, depth=1, n2q=0, age=0),
    ]
    e = elitism(pop, 1)
    assert e[0].fitness == 2.0
