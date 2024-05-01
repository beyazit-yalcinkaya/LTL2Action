import ring
import numpy as np

import dgl
import networkx as nx
from copy import deepcopy
from pysat.solvers import Solver

feature_inds = {"rejecting": -1, "accepting": -2, "temp": -3, "normal": -4, "init": -5, "AND": -6, "OR": -7}

"""
A class that can take an DFA formula and generate the Abstract Syntax Tree (DFA) of it. This
code can generate trees in either Networkx or DGL formats. And uses caching to remember recently
generated trees.
"""
class DFABuilder(object):
    def __init__(self, propositions):
        super(DFABuilder, self).__init__()
        self.propositions = propositions
        self.feature_size = len(self.propositions) + len(feature_inds)

    # To make the caching work.
    def __ring_key__(self):
        return "DFABuilder"

    def __call__(self, dfa_goal, library="dgl"):
        return self._to_graph(dfa_goal, library)

    @ring.lru(maxsize=400000)
    def _to_graph(self, dfa_goal, library="dgl"):
        from utils.env import edge_types
        cnf_nxgs = []
        cnf_or_nodes = []
        cnf_rename = []
        for i, dfa_clause in enumerate(dfa_goal):
            clause_nxgs = []
            clause_init_nodes = []
            clause_rename = []
            for j, dfa in enumerate(dfa_clause):
                clause_nxg, clause_init_node = self.dfa_to_formatted_nxg(dfa)
                clause_nxgs.append(clause_nxg)
                clause_rename.append(str(j) + "_")
                clause_init_nodes.append(str(j) + "_" + clause_init_node)
            if len(clause_nxgs) > 1:
                composed_clause_nxg = nx.union_all(clause_nxgs, rename=clause_rename)
            elif len(clause_nxgs) == 1:
                composed_clause_nxg = clause_nxgs[0]
                nx.relabel_nodes(composed_clause_nxg, {node: str(j) + "_" + node for node in composed_clause_nxg.nodes}, copy=False)
            else:
                raise NotImplemented
            or_node = "OR"
            composed_clause_nxg.add_node(or_node, feat=np.array([[0.0] * self.feature_size]))
            composed_clause_nxg.nodes[or_node]["feat"][0][feature_inds["OR"]] = 1.0
            for clause_init_node in clause_init_nodes:
                composed_clause_nxg.add_edge(clause_init_node, or_node, type=edge_types["OR"])
            cnf_nxgs.append(composed_clause_nxg)
            cnf_rename.append(str(i) + "_")
            cnf_or_nodes.append(str(i) + "_" + or_node)
        if len(cnf_nxgs) > 1:
            composed_cnf_nxg = nx.union_all(cnf_nxgs, rename=cnf_rename)
        elif len(cnf_nxgs) == 1:
            composed_cnf_nxg = cnf_nxgs[0]
            nx.relabel_nodes(composed_cnf_nxg, {node: str(i) + "_" + node for node in composed_cnf_nxg.nodes}, copy=False)
        else:
            raise NotImplemented
        nx.set_node_attributes(composed_cnf_nxg, np.array([0.0], dtype=np.float32), "is_root")
        and_node = "AND"
        composed_cnf_nxg.add_node(and_node, feat=np.array([[0.0] * self.feature_size]), is_root=np.array([1.0], dtype=np.float32))
        composed_cnf_nxg.nodes[and_node]["feat"][0][feature_inds["AND"]] = 1.0
        for cnf_or_node in cnf_or_nodes:
            composed_cnf_nxg.add_edge(cnf_or_node, and_node, type=edge_types["AND"])
        for node in composed_cnf_nxg.nodes:
            composed_cnf_nxg.add_edge(node, node, type=edge_types["self"])
        nxg = composed_cnf_nxg

        if (library == "networkx"): return nxg

        # convert the Networkx graph to dgl graph and pass the 'feat' attribute
        g = dgl.DGLGraph()
        g.from_networkx(nxg, node_attrs=["feat", "is_root"], edge_attrs=["type"]) # dgl does not support string attributes (i.e., token)
        return g

    @ring.lru(maxsize=600000)
    def dfa_to_formatted_nxg(self, dfa):
        from utils.env import edge_types

        nxg = nx.DiGraph()
        new_node_name_counter = 0
        new_node_name_base_str = "temp_"

        for s in dfa.states():
            start = str(s)
            nxg.add_node(start)
            nxg.nodes[start]["feat"] = np.array([[0.0] * self.feature_size])
            nxg.nodes[start]["feat"][0][feature_inds["normal"]] = 1.0
            if dfa._label(s): # is accepting?
                nxg.nodes[start]["feat"][0][feature_inds["accepting"]] = 1.0
            elif sum(s != dfa._transition(s, a) for a in dfa.inputs) == 0: # is rejecting?
                nxg.nodes[start]["feat"][0][feature_inds["rejecting"]] = 1.0
            embeddings = {}
            for a in dfa.inputs:
                e = dfa._transition(s, a)
                if s == e:
                    continue # We define self loops later when composing graphs
                end = str(e)
                if end not in embeddings.keys():
                    embeddings[end] = np.zeros(self.feature_size)
                    embeddings[end][feature_inds["temp"]] = 1.0 # Since it is a temp node
                embeddings[end][self.propositions.index(a)] = 1.0
            for end in embeddings.keys():
                new_node_name = new_node_name_base_str + str(new_node_name_counter)
                new_node_name_counter += 1
                nxg.add_node(new_node_name, feat=np.array([embeddings[end]]))
                nxg.add_edge(end, new_node_name, type=edge_types["normal-to-temp"])
                nxg.add_edge(new_node_name, start, type=edge_types["temp-to-normal"])

        init_node = str(dfa.start)
        nxg.nodes[init_node]["feat"][0][feature_inds["init"]] = 1.0

        return nxg, init_node

def draw(G, formula):
    from networkx.drawing.nx_agraph import graphviz_layout
    import matplotlib.pyplot as plt

    colors = ["black", "red", "green", "blue", "purple", "orange"]
    edge_color = [colors[i] for i in nx.get_edge_attributes(G,'type').values()]

    plt.title(formula)
    pos=graphviz_layout(G, prog='dot')
    # labels = nx.get_node_attributes(G,'token')
    labels = G.nodes
    nx.draw(G, pos, with_labels=True, arrows=True, node_shape='s', edgelist=list(nx.get_edge_attributes(G,'type')), node_size=500, node_color="white", edge_color=edge_color) #edge_color=edge_color
    plt.show()

"""
A simple test to check if the DFABuilder works fine. We do a preorder DFS traversal of the resulting
tree and convert it to a simplified formula and compare the result with the simplified version of the
original formula. They should match.
"""
if __name__ == '__main__':
    import re
    import sys
    import itertools
    import matplotlib.pyplot as plt

    sys.path.insert(0, '../../')
    from dfa_samplers import getDFASampler

    for sampler_id, _ in itertools.product(["Default", "Sequence_2_20"], range(20)):
        props = "abcdefghijklmnopqrst"
        sampler = getDFASampler(sampler_id, props)
        builder = DFABuilder(list(set(list(props))))
        formula = sampler.sample()
        tree = builder(formula, library="networkx")
        pre = list(nx.dfs_preorder_nodes(tree, source=0))
        draw(tree, formula)
        u_tree = tree.to_undirected()
        pre = list(nx.dfs_preorder_nodes(u_tree, source=0))

        original = re.sub('[,\')(]', '', str(formula))
        observed = " ".join([u_tree.nodes[i]["token"] for i in pre])

        assert original == observed, f"Test Faield: Expected: {original}, Got: {observed}"

    print("Test Passed!")
