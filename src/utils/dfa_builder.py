import ring
import numpy as np

import dgl
import networkx as nx
from dfa import dfa2dict
from copy import deepcopy
from pysat.solvers import Solver

FEATURE_SIZE = 22 # TODO: Fix this

feature_inds = {"rejecting": -1, "accepting": -2, "temp": -3, "normal": -4, "init": -5, "AND": -6, "OR": -7, "NOP": -8}

"""
A class that can take an DFA formula and generate the Abstract Syntax Tree (DFA) of it. This
code can generate trees in either Networkx or DGL formats. And uses caching to remember recently
generated trees.
"""
class DFABuilder(object):
    def __init__(self, propositions):
        super(DFABuilder, self).__init__()

        self.propositions = propositions

    # To make the caching work.
    def __ring_key__(self):
        return "DFABuilder"

    def __call__(self, dfas, library="dgl"):
        op, dfas = dfas
        assert op in ["AND", "OR", "NOP"]
        dfas = tuple(dfa2dict(dfa) for dfa in dfas)
        return self._to_graph(op, dfas, library)

    @ring.lru(maxsize=1000000)
    def _to_graph(self, op, dfas, library="dgl"):
        from utils.env import edge_types
        nxgs = []
        init_nodes = []
        for i, (dfa_dict, init_state) in enumerate(dfas):
            nxg_i, init_node = self.get_nxg_from_dfa_dict(dfa_dict, init_state)
            nxg_i = nxg_i.reverse(copy=True)
            nxg_i = nx.relabel_nodes(nxg_i, lambda x: str(i) + "_" + x, copy=True)
            nxgs.append(nxg_i)
            init_nodes.append(str(i) + "_" + init_node)

        nxg = nx.compose_all(nxgs)
        nx.set_node_attributes(nxg, 0.0, "is_root")
        nxg.add_node(op, feat=np.array([[0.0] * FEATURE_SIZE]), is_root=1.0)
        nxg.nodes[op]["feat"][0][feature_inds[op]] = 1.0
        for init_node in init_nodes:
            nxg.add_edge(init_node, op, type=edge_types[op])

        for node in nxg.nodes:
            nxg.add_edge(node, node, type=edge_types["self"])

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
        onehot_embedding[feature_inds["temp"]] = 1.0 # Since it will be a temp node
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

    def dfa2nxg(self, dfa_dict, init_node, minimize=False):

        init_node = str(init_node)

        nxg = nx.DiGraph()

        accepting_states = []
        for start, (accepting, transitions) in dfa_dict.items():
            start = str(start)
            nxg.add_node(start)
            if accepting:
                accepting_states.append(start)
            for action, end in transitions.items():
                if nxg.has_edge(start, str(end)):
                    existing_label = nxg.get_edge_data(start, str(end))['label']
                    nxg.add_edge(start, str(end), label='{} | {}'.format(existing_label, action))
                else:
                    nxg.add_edge(start, str(end), label=action)

        # nxg = nx.ego_graph(nxg, init_node, radius=5)
        accepting_states = list(set(accepting_states).intersection(set(nxg.nodes)))

        return init_node, accepting_states, nxg


    def _format(self, init_node, accepting_states, nxg):
        from utils.env import edge_types
        rejecting_states = []
        for node in nxg.nodes:
            if self._is_sink_state(node, nxg) and node not in accepting_states:
                rejecting_states.append(node)

        for node in nxg.nodes:
            nxg.nodes[node]["feat"] = np.array([[0.0] * FEATURE_SIZE])
            nxg.nodes[node]["feat"][0][feature_inds["normal"]] = 1.0
            if node in accepting_states:
                nxg.nodes[node]["feat"][0][feature_inds["accepting"]] = 1.0
            if node in rejecting_states:
                nxg.nodes[node]["feat"][0][feature_inds["rejecting"]] = 1.0

        nxg.nodes[init_node]["feat"][0][feature_inds["init"]] = 1.0

        edges = deepcopy(nxg.edges)

        new_node_name_base_str = "temp_"
        new_node_name_counter = 0

        for e in edges:
            guard = nxg.edges[e]["label"]
            nxg.remove_edge(*e)
            if e[0] == e[1]:
                continue # We define self loops below
            onehot_embedding = self._get_onehot_guard_embeddings(guard) # It is ok if we receive a cached embeddding since we do not modify it
            if len(onehot_embedding) == 0:
                continue
            new_node_name = new_node_name_base_str + str(new_node_name_counter)
            new_node_name_counter += 1
            nxg.add_node(new_node_name, feat=np.array(onehot_embedding))
            nxg.add_edge(e[0], new_node_name, type=edge_types["normal-to-temp"])
            nxg.add_edge(new_node_name, e[1], type=edge_types["temp-to-normal"])

        return nxg, init_node

    def get_nxg_from_dfa_dict(self, dfa_dict, init_state):
        init_node, accepting_states, nxg = self.dfa2nxg(dfa_dict, init_state)
        dfa_nxg, init_node = self._format(init_node, accepting_states, nxg)
        return dfa_nxg, init_node


def draw(G, formula):
    from networkx.drawing.nx_agraph import graphviz_layout
    import matplotlib.pyplot as plt

    # colors = ["black", "red"]
    # edge_color = [colors[i] for i in nx.get_edge_attributes(G,'type').values()]

    plt.title(formula)
    pos=graphviz_layout(G, prog='dot')
    # labels = nx.get_node_attributes(G,'token')
    labels = G.nodes
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
