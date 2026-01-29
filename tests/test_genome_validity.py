from qarchga.genomes import Genome

def test_depth_nonnegative():
    g = Genome(n_qubits=3, layers=[[]])
    assert g.depth() == 1
