import numpy as np

import dgl
import networkx as nx
from dfa import dfa2dict
from copy import deepcopy
from pysat.solvers import Solver

edge_types = {k:v for (v, k) in enumerate(["self", "arg", "arg1", "arg2"])}

FEATURE_SIZE = 22

"""
A class that can take an DFA formula and generate the Abstract Syntax Tree (DFA) of it. This
code can generate trees in either Networkx or DGL formats. And uses caching to remember recently
generated trees.
"""
class DFABuilder(object):
    def __init__(self, propositions):
        super(DFABuilder, self).__init__()

        self.propositions = propositions

    def __call__(self, dfa, library="dgl"):
        nxg = self.get_nxg_from_dfa(dfa)

        if (library == "networkx"): return nxg

        # convert the Networkx graph to dgl graph and pass the 'feat' attribute
        g = dgl.DGLGraph()
        g.from_networkx(nxg, node_attrs=["feat", "is_root"], edge_attrs=["type"]) # dgl does not support string attributes (i.e., token)
        return g

    def _get_guard_embeddings(self, guard):
        embeddings = []
        try:
            guard = guard.replace(" ", "").replace("(", "").replace(")", "").replace("\"", "")
        except:
            return embeddings
        if (guard == "true"):
            return embeddings
        guard = guard.split("&")
        cnf = []
        seen_atoms = []
        # print("guard", guard)
        for c in guard:
            atoms = c.split("|")
            clause = []
            for atom in atoms:
                try:
                    index = seen_atoms.index(atom if atom[0] != "~" else atom[1:])
                except:
                    index = len(seen_atoms)
                    seen_atoms.append(atom if atom[0] != "~" else atom[1:])
                clause.append(index + 1 if atom[0] != "~" else -(index + 1))
            cnf.append(clause)
        models = []
        with Solver(bootstrap_with=cnf) as s:
            models = list(s.enum_models())
        if len(models) == 0:
            return embeddings
        for model in models:
            temp = [0.0] * FEATURE_SIZE
            for a in model:
                if a > 0:
                    atom = seen_atoms[abs(a) - 1]
                    temp[self.propositions.index(atom)] = 1.0
            embeddings.append(temp)
        return embeddings

    def _get_onehot_guard_embeddings(self, guard):
        is_there_onehot = False
        is_there_all_zero = False
        onehot_embedding = [0.0] * FEATURE_SIZE
        onehot_embedding[-3] = 1.0 # Since it will be a temp node
        full_embeddings = self._get_guard_embeddings(guard)
        for embed in full_embeddings:
            # discard all non-onehot embeddings (a one-hot embedding must contain only a single 1)
            if embed.count(1.0) == 1:
                # clean the embedding so that it's one-hot
                is_there_onehot = True
                var_idx = embed.index(1.0)
                onehot_embedding[var_idx] = 1.0
            elif embed.count(0.0) == len(embed):
                is_there_all_zero = True
        if is_there_onehot or is_there_all_zero:
            return [onehot_embedding]
        else:
            return []

    def _is_sink_state(self, node, nxg):
        for edge in nxg.edges:
            if node == edge[0] and node != edge[1]: # If there is an outgoing edge to another node, then it is not an accepting state
                return False
        return True

    def dfa2nxg(self, mvc_dfa, minimize=False):
        """ converts a mvc format dfa into a networkx dfa """

        dfa_dict, init_node = dfa2dict(mvc_dfa)
        init_node = str(init_node)

        nxg = nx.DiGraph()

        accepting_states = []
        for start, (accepting, transitions) in dfa_dict.items():
            # pydot_graph.add_node(nodes[start])
            start = str(start)
            nxg.add_node(start)
            if accepting:
                accepting_states.append(start)
            for action, end in transitions.items():
                if nxg.has_edge(start, str(end)):
                    existing_label = nxg.get_edge_data(start, str(end))['label']
                    nxg.add_edge(start, str(end), label='{} | {}'.format(existing_label, action))
                    # print('{} | {}'.format(existing_label, action))
                else:
                    nxg.add_edge(start, str(end), label=action)

        return init_node, accepting_states, nxg


    def _format(self, init_node, accepting_states, nxg):
        # print('init', init_node)
        # print('accepting', accepting_states)
        rejecting_states = []
        for node in nxg.nodes:
            if self._is_sink_state(node, nxg) and node not in accepting_states:
                rejecting_states.append(node)

        for node in nxg.nodes:
            nxg.nodes[node]["feat"] = np.array([[0.0] * FEATURE_SIZE])
            nxg.nodes[node]["feat"][0][-4] = 1.0
            if node in accepting_states:
                nxg.nodes[node]["feat"][0][-2] = 1.0
            if node in rejecting_states:
                nxg.nodes[node]["feat"][0][-1] = 1.0

        nxg.nodes[init_node]["feat"][0][-5] = 1.0

        edges = deepcopy(nxg.edges)

        new_node_name_base_str = "temp_"
        new_node_name_counter = 0

        for e in edges:
            # print(e, nxg.edges[e])
            guard = nxg.edges[e]["label"]
            # print(e, guard)
            nxg.remove_edge(*e)
            if e[0] == e[1]:
                continue # We define self loops below
            onehot_embedding = self._get_onehot_guard_embeddings(guard) # It is ok if we receive a cached embeddding since we do not modify it
            if len(onehot_embedding) == 0:
                continue
            new_node_name = new_node_name_base_str + str(new_node_name_counter)
            new_node_name_counter += 1
            nxg.add_node(new_node_name, feat=np.array(onehot_embedding))
            nxg.add_edge(e[0], new_node_name, type=2)
            nxg.add_edge(new_node_name, e[1], type=3)

        nx.set_node_attributes(nxg, 0.0, "is_root")
        nxg.nodes[init_node]["is_root"] = 1.0 # is_root means current state

        for node in nxg.nodes:
            nxg.add_edge(node, node, type=1)

        return nxg

    def get_nxg_from_dfa(self, dfa):
        init_node, accepting_states, nxg = self.dfa2nxg(dfa)
        dfa_nxg = self._format(init_node,accepting_states,nxg)
        return dfa_nxg


def draw(G, formula):
    from networkx.drawing.nx_agraph import graphviz_layout
    import matplotlib.pyplot as plt

    # colors = ["black", "red"]
    # edge_color = [colors[i] for i in nx.get_edge_attributes(G,'type').values()]

    plt.title(formula)
    pos=graphviz_layout(G, prog='dot')
    labels = nx.get_node_attributes(G,'token')
    nx.draw(G, pos, with_labels=True, arrows=True, labels=labels, node_shape='s', edgelist=list(nx.get_edge_attributes(G,'type')), node_size=500, node_color="white") #edge_color=edge_color
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
