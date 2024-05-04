"""
This class is responsible for sampling DFA formulas typically from
given template(s).

@ propositions: The set of propositions to be used in the sampled
                formula at random.
"""

import random
from dfa import DFA

class DFASampler():
    def __init__(self, propositions):
        self.propositions = propositions

    def sample(self):
        raise NotImplementedError


# Samples from one of the other samplers at random. The other samplers are sampled by their default args.
class SuperSampler(DFASampler):
    def __init__(self, propositions):
        super().__init__(propositions)
        self.reg_samplers = getRegisteredSamplers(self.propositions)

    def sample(self):
        return random.choice(self.reg_samplers).sample()

# This class samples formulas of form (or, op_1, op_2), where op_1 and 2 can be either specified as samplers_ids
# or by default they will be sampled at random via SuperSampler.
class OrSampler(DFASampler):
    def __init__(self, propositions, sampler_ids = ["SuperSampler"]*2):
        super().__init__(propositions)
        self.sampler_ids = sampler_ids

    def sample(self):
        return ('or', getDFASampler(self.sampler_ids[0], self.propositions).sample(),
                        getDFASampler(self.sampler_ids[1], self.propositions).sample())
class JoinSampler(DFASampler):
    def __init__(self, propositions, sampler_ids):
        super().__init__(propositions)
        self.n = len(sampler_ids)
        self.samplers = [getDFASampler(sampler_id, self.propositions) for sampler_id in sampler_ids]

    def sample(self):
        return random.choice(self.samplers).sample()

# This class generates random LTL formulas using the following template:
#   ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))
# where p1, p2, p3, and p4 are randomly sampled propositions
class DefaultSampler(DFASampler):
    def sample(self):
        p = random.sample(self.propositions,4)
        return ('until',('not',p[0]),('and', p[1], ('until',('not',p[2]),p[3])))

# This class generates random conjunctions of Until-Tasks.
# Each until tasks has *n* levels, where each level consists
# of avoiding a proposition until reaching another proposition.
#   E.g.,
#      Level 1: ('until',('not','a'),'b')
#      Level 2: ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))
#      etc...
# The number of until-tasks, their levels, and their propositions are randomly sampled.
# This code is a generalization of the DefaultSampler---which is equivalent to UntilTaskSampler(propositions, 2, 2, 1, 1)
class UntilTaskSampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        assert 2*int(max_levels)*int(max_conjunctions) <= len(propositions), "The domain does not have enough propositions!"

    def sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        p = random.sample(self.propositions,2*self.levels[1]*n_conjs)
        ltl = None
        seqs = []
        b = 0
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            # Sampling an until task of *n_levels* levels
            until_task = ('until',('not',p[b]),p[b+1])
            seq = [(p[b], p[b+1])]
            b +=2
            for j in range(1,n_levels):
                until_task = ('until',('not',p[b]),('and', p[b+1], until_task))
                seq = [(p[b], p[b+1])] + seq
                b +=2
            # Adding the until task to the conjunction of formulas that the agent have to solve
            if ltl is None: ltl = until_task
            else:           ltl = ('and',until_task,ltl)
            seqs = [tuple(seq)] + seqs
        seqs = tuple(seqs)
        def delta(s, c):
            if s is not None:
                for i in range(len(s)):
                    if s[i] != () and c != s[i][0][0] and c == s[i][0][1]:
                        return s[:i] + (s[i][1:],) + s[i + 1:]
                    elif s[i] != () and c == s[i][0][0]:
                        return None
            return s
        return ((DFA(
            start=seqs,
            inputs=self.propositions,
            label=lambda s: s == tuple(tuple() for _ in range(n_conjs)),
            transition=delta,
        ),),)

class CompositionalUntilTaskSampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        assert 2*int(max_levels)*int(max_conjunctions) <= len(propositions), "The domain does not have enough propositions!"

    def sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        p = random.sample(self.propositions,2*self.levels[1]*n_conjs)
        ltl = None
        seqs = []
        b = 0
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            # Sampling an until task of *n_levels* levels
            until_task = ('until',('not',p[b]),p[b+1])
            seq = [(p[b], p[b+1])]
            b +=2
            for j in range(1,n_levels):
                until_task = ('until',('not',p[b]),('and', p[b+1], until_task))
                seq = [(p[b], p[b+1])] + seq
                b +=2
            # Adding the until task to the conjunction of formulas that the agent have to solve
            if ltl is None: ltl = until_task
            else:           ltl = ('and',until_task,ltl)
            seqs = [tuple(seq)] + seqs
        seqs = tuple(seqs)
        def delta(s, c):
            if s is not None:
                if s != () and c != s[0][0] and c == s[0][1]:
                    return s[1:]
                elif s != () and c == s[0][0]:
                    return None
            return s
        dfas = tuple(DFA(start=seq, inputs=self.propositions, label=lambda s: s == tuple(), transition=delta) for seq in seqs)
        return tuple((dfa,) for dfa in dfas)

    # def sample(self):
    #     conjs = random.randint(*self.conjunctions)
    #     seqs = tuple(self.sample_sequence() for _ in range(conjs))
    #     dfas = tuple(DFA(start=seq, inputs=self.propositions, label=lambda s: s == tuple(), transition=lambda s, c: s[1:] if s != () and c != s[0][0] and c == s[0][1] else s) for seq in seqs)
    #     return tuple((dfa,) for dfa in dfas)


# This class generates random LTL formulas that form a sequence of actions.
# @ min_len, max_len: min/max length of the random sequence to generate.
class SequenceSampler(DFASampler):
    def __init__(self, propositions, min_len=2, max_len=4):
        super().__init__(propositions)
        self.min_len = int(min_len)
        self.max_len = int(max_len)

    def sample(self):
        length = random.randint(self.min_len, self.max_len)
        seq = ""

        while len(seq) < length:
            c = random.choice(self.propositions)
            if len(seq) == 0 or seq[-1] != c:
                seq += c

        ret = self._get_sequence(seq)

        return ret

    def _get_sequence(self, seq):
        if len(seq) == 1:
            return ('eventually',seq)
        return ('eventually',('and', seq[0], self._get_sequence(seq[1:])))

# This generates several sequence tasks which can be accomplished in parallel. 
# e.g. in (eventually (a and eventually c)) and (eventually b)
# the two sequence tasks are "a->c" and "b".
class EventuallySampler(DFASampler):
    def __init__(self, propositions, min_levels = 1, max_levels=4, min_conjunctions=1, max_conjunctions=3):
        super().__init__(propositions)
        assert(len(propositions) >= 3)
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        self.levels = (int(min_levels), int(max_levels))

    def sample(self):
        conjs = random.randint(*self.conjunctions)
        seqs = tuple(self.sample_sequence() for _ in range(conjs))
        def delta(s, c):
            for i in range(len(s)):
                if s[i] != () and c in s[i][0]:
                    return s[:i] + (s[i][1:],) + s[i + 1:]
            return s
        return ((DFA(
            start=seqs,
            inputs=self.propositions,
            label=lambda s: s == tuple(tuple() for _ in range(conjs)),
            transition=delta,
        ),),)

    def sample_sequence(self):
        length = random.randint(*self.levels)
        seq = []

        last = []
        while len(seq) < length:
            # Randomly replace some propositions with a disjunction to make more complex formulas
            population = [p for p in self.propositions if p not in last]

            if random.random() < 0.25:
                c = random.sample(population, 2)
            else:
                c = random.sample(population, 1)

            seq.append(tuple(c))
            last = c

        return tuple(seq)

class CompositionalEventuallySampler(EventuallySampler):
    def __init__(self, propositions, min_levels = 1, max_levels=4, min_conjunctions=1, max_conjunctions=3):
        super().__init__(propositions, min_levels, max_levels, min_conjunctions, max_conjunctions)
        assert(len(propositions) >= 3)
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        self.levels = (int(min_levels), int(max_levels))

    def sample(self):
        conjs = random.randint(*self.conjunctions)
        seqs = tuple(self.sample_sequence() for _ in range(conjs))
        dfas = tuple(DFA(start=seq, inputs=self.propositions, label=lambda s: s == tuple(), transition=lambda s, c: s[1:] if s != () and c in s[0] else s) for seq in seqs)
        return tuple((dfa,) for dfa in dfas)


class AdversarialEnvSampler(DFASampler):
    def sample(self):
        p = random.randint(0,1)
        if p == 0:
            def delta(s, c):
                if s == 0 and c == 'a':
                    return 1
                elif s == 1 and c == 'b':
                    return 2
                return s
            return ((DFA(
                start=0,
                inputs=self.propositions,
                label=lambda s: s == 2,
                transition=delta,
            ),),)
        else:
            def delta(s, c):
                if s == 0 and c == 'a':
                    return 1
                elif s == 1 and c == 'c':
                    return 2
                return s
            return ((DFA(
                start=0,
                inputs=self.propositions,
                label=lambda s: s == 2,
                transition=delta,
            ),),)

class ParitySampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        assert 3*int(max_levels)*int(max_conjunctions) <= len(propositions), "The domain does not have enough propositions!"

    def sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        p = random.sample(self.propositions,3*self.levels[1]*n_conjs)
        ltl = None
        seqs = []
        b = 0
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            seq = [(p[b], p[b + 1], p[b + 2])]
            b += 3
            for j in range(1, n_levels):
                seq = [(p[b], p[b + 1], p[b + 2])] + seq
                b += 3
            seqs = [tuple(seq)] + seqs
        seqs = tuple(seqs)
        def delta(s, c):
            s, is_in_recovery_mode = s
            if is_in_recovery_mode:
                for i in range(len(s)):
                    if s[i] != () and c == s[i][0][2]: # Fix
                        return s, False
            else:
                for i in range(len(s)):
                    if s[i] != () and c == s[i][0][0]: # Reach
                        return s[:i] + (s[i][1:],) + s[i + 1:], False
                    elif s[i] != () and c == s[i][0][1]: # Avoid
                        return s, True
            return s, is_in_recovery_mode
        return ((DFA(
            start=(seqs, False),
            inputs=self.propositions,
            label=lambda s: s[0] == tuple(tuple() for _ in range(n_conjs)) and not s[1],
            transition=delta,
        ),),)

class CompositionalParitySampler(DFASampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        assert 3*int(max_levels)*int(max_conjunctions) <= len(propositions), "The domain does not have enough propositions!"

    def sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        p = random.sample(self.propositions,3*self.levels[1]*n_conjs)
        ltl = None
        seqs = []
        b = 0
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            seq = [(p[b], p[b + 1], p[b + 2])]
            b += 3
            for j in range(1, n_levels):
                seq = [(p[b], p[b + 1], p[b + 2])] + seq
                b += 3
            seqs = [tuple(seq)] + seqs
        seqs = tuple(seqs)
        def delta(s, c):
            is_in_recovery_mode, s = s
            if is_in_recovery_mode:
                if s != () and c == s[0][2]: # Fix
                    return False, s
            else:
                if s != () and c == s[0][0]: # Reach
                    return False, s[1:]
                elif s != () and c == s[0][1]: # Avoid
                    return True, s
            return is_in_recovery_mode, s
        dfas = tuple(DFA(start=(False, seq), inputs=self.propositions, label=lambda s: not s[0] and s[1] == tuple(), transition=delta) for seq in seqs)
        return tuple((dfa,) for dfa in dfas)

def getRegisteredSamplers(propositions):
    return [SequenceSampler(propositions),
            UntilTaskSampler(propositions),
            DefaultSampler(propositions),
            EventuallySampler(propositions),
            CompositionalEventuallySampler(propositions)]

# The DFASampler factory method that instantiates the proper sampler
# based on the @sampler_id.
def getDFASampler(sampler_id, propositions):
    tokens = ["Default"]
    if (sampler_id != None):
        tokens = sampler_id.split("_")

    # Don't change the order of ifs here otherwise the OR sampler will fail
    if (tokens[0] == "OrSampler"):
        return OrSampler(propositions)
    elif ("_OR_" in sampler_id): # e.g., Sequence_2_4_OR_UntilTask_3_3_1_1
        sampler_ids = sampler_id.split("_OR_")
        return OrSampler(propositions, sampler_ids)
    elif ("_JOIN_" in sampler_id): # e.g., Eventually_1_5_1_4_JOIN_Until_1_3_1_2
        sampler_ids = sampler_id.split("_JOIN_")
        return JoinSampler(propositions, sampler_ids)
    elif (tokens[0] == "Parity"):
        return ParitySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "CompositionalParity"):
        return CompositionalParitySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "Sequence"):
        return SequenceSampler(propositions, tokens[1], tokens[2])
    elif (tokens[0] == "Until"):
        return UntilTaskSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "CompositionalUntil"):
        return CompositionalUntilTaskSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "SuperSampler"):
        return SuperSampler(propositions)
    elif (tokens[0] == "Adversarial"):
        return AdversarialEnvSampler(propositions)
    elif (tokens[0] == "Eventually"):
        return EventuallySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "CompositionalEventually"):
        return CompositionalEventuallySampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    else: # "Default"
        return DefaultSampler(propositions)

