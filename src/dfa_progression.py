"""
This code allows to progress DFA formulas. It requires installing the SPOT library:
    - https://spot.lrde.epita.fr/install.html
To encode DFA formulas, we use tuples, e.g.,
    (
        'and',
        ('until','True', ('and', 'd', ('until','True','c'))),
        ('until','True', ('and', 'a', ('until','True', ('and', 'b', ('until','True','c')))))
    )
Some notes about the format:
    - It supports the following temporal operators: "next", "until", "always", and "eventually".
    - It supports the following logical operators: "not", "or", "and".
    - Propositions are assume to be one char.
    - Negations are always followed by a proposition.
    - true and false are encoded as "True" and "False"
"""


from sympy import *
from sympy.logic import simplify_logic
from sympy.logic.boolalg import And, Or, Not
import time, collections, spot

"""
This module contains functions to progress co-safe DFA formulas such as:
    (
        'and',
        ('until','True', ('and', 'd', ('until','True','c'))),
        ('until','True', ('and', 'a', ('until','True', ('and', 'b', ('until','True','c')))))
    )
"""


def _is_prop_formula(f):
    # returns True if the formula does not contains temporal operators
    return 'next' not in str(f) and 'until' not in str(f)

def _subsume_until(f1, f2):
    if str(f1) not in str(f2):
        return False
    while type(f2) != str:
        if f1 == f2:
            return True
        if f2[0] == 'until':
            f2 = f2[2]
        elif f2[0] == 'and':
            if _is_prop_formula(f2[1]) and not _is_prop_formula(f2[2]):
                f2 = f2[2]
            elif not _is_prop_formula(f2[1]) and _is_prop_formula(f2[2]):
                f2 = f2[1]
            else:
                return False
        else:
            return False
    return False

def _subsume_or(f1, f2):
    if str(f1) not in str(f2):
        return False
    while type(f2) != str:
        if f1 == f2:
            return True
        if f2[0] == 'until':
            f2 = f2[2]
        elif f2[0] == 'and':
            if _is_prop_formula(f2[1]) and not _is_prop_formula(f2[2]):
                f2 = f2[2]
            elif not _is_prop_formula(f2[1]) and _is_prop_formula(f2[2]):
                f2 = f2[1]
            else:
                return False
        else:
            return False
    return False


def progress_and_clean(dfa_formula, truth_assignment):
    dfa = progress(dfa_formula, truth_assignment)
    # I am using spot to simplify the resulting dfa formula
    dfa_spot = _get_spot_format(dfa)
    f = spot.formula(dfa_spot)
    f = spot.simplify(f)
    dfa_spot = f.__format__("l")
    dfa_std,r = _get_std_format(dfa_spot.split(' '))
    assert len(r) == 0, "Format error" + str(dfa_std) + " " + str(r)
    return dfa_std


def spotify(dfa_formula):
    dfa_spot = _get_spot_format(dfa_formula)
    f = spot.formula(dfa_spot)
    f = spot.simplify(f)
    dfa_spot = f.__format__("l")
    # return dfa_spot
    return f#.to_str('latex')


def _get_spot_format(dfa_std):
    dfa_spot = str(dfa_std).replace("(","").replace(")","").replace(",","")
    dfa_spot = dfa_spot.replace("'until'","U").replace("'not'","!").replace("'or'","|").replace("'and'","&")
    dfa_spot = dfa_spot.replace("'next'","X").replace("'eventually'","F").replace("'always'","G").replace("'True'","t").replace("'False'","f").replace("\'","\"")
    return dfa_spot

def _get_std_format(dfa_spot):

    s = dfa_spot[0]
    r = dfa_spot[1:]

    if s in ["X","U","&","|"]:
        v1,r1 = _get_std_format(r)
        v2,r2 = _get_std_format(r1)
        if s == "X": op = 'next'
        if s == "U": op = 'until'
        if s == "&": op = 'and'
        if s == "|": op = 'or'
        return (op,v1,v2),r2

    if s in ["F","G","!"]:
        v1,r1 = _get_std_format(r)
        if s == "F": op = 'eventually'
        if s == "G": op = 'always'
        if s == "!": op = 'not'
        return (op,v1),r1

    if s == "f":
        return 'False', r

    if s == "t":
        return 'True', r

    if s[0] == '"':
        return s.replace('"',''), r

    assert False, "Format error in spot2std"

def progress(dfa_formula, truth_assignment):
    if type(dfa_formula) == str:
        # True, False, or proposition
        if len(dfa_formula) == 1:
            # dfa_formula is a proposition
            if dfa_formula in truth_assignment:
                return 'True'
            else:
                return 'False'
        return dfa_formula

    if dfa_formula[0] == 'not':
        # negations should be over propositions only according to the cosafe dfa syntactic restriction
        result = progress(dfa_formula[1], truth_assignment)
        if result == 'True':
            return 'False'
        elif result == 'False':
            return 'True'
        else:
            raise NotImplementedError("The following formula doesn't follow the cosafe syntactic restriction: " + str(dfa_formula))

    if dfa_formula[0] == 'and':
        res1 = progress(dfa_formula[1], truth_assignment)
        res2 = progress(dfa_formula[2], truth_assignment)
        if res1 == 'True' and res2 == 'True': return 'True'
        if res1 == 'False' or res2 == 'False': return 'False'
        if res1 == 'True': return res2
        if res2 == 'True': return res1
        if res1 == res2:   return res1
        #if _subsume_until(res1, res2): return res2
        #if _subsume_until(res2, res1): return res1
        return ('and',res1,res2)

    if dfa_formula[0] == 'or':
        res1 = progress(dfa_formula[1], truth_assignment)
        res2 = progress(dfa_formula[2], truth_assignment)
        if res1 == 'True'  or res2 == 'True'  : return 'True'
        if res1 == 'False' and res2 == 'False': return 'False'
        if res1 == 'False': return res2
        if res2 == 'False': return res1
        if res1 == res2:    return res1
        #if _subsume_until(res1, res2): return res1
        #if _subsume_until(res2, res1): return res2
        return ('or',res1,res2)

    if dfa_formula[0] == 'next':
        return progress(dfa_formula[1], truth_assignment)

    # NOTE: What about release and other temporal operators?
    if dfa_formula[0] == 'eventually':
        res = progress(dfa_formula[1], truth_assignment)
        return ("or", dfa_formula, res)

    if dfa_formula[0] == 'always':
        res = progress(dfa_formula[1], truth_assignment)
        return ("and", dfa_formula, res)

    if dfa_formula[0] == 'until':
        res1 = progress(dfa_formula[1], truth_assignment)
        res2 = progress(dfa_formula[2], truth_assignment)

        if res1 == 'False':
            f1 = 'False'
        elif res1 == 'True':
            f1 = ('until', dfa_formula[1], dfa_formula[2])
        else:
            f1 = ('and', res1, ('until', dfa_formula[1], dfa_formula[2]))

        if res2 == 'True':
            return 'True'
        if res2 == 'False':
            return f1

        # Returning ('or', res2, f1)
        #if _subsume_until(f1, res2): return f1
        #if _subsume_until(res2, f1): return res2
        return ('or', res2, f1)


if __name__ == '__main__':
    #dfa = ('and',('eventually','a'),('and',('eventually','b'),('eventually','c')))
    #dfa = ('and',('eventually','a'),('eventually',('and','b',('eventually','c'))))
    #dfa = ('until',('not','a'),('and', 'b', ('eventually','d')))
    dfa = ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))

    while True:
        print(dfa)
        props = input()
        dfa = progress_and_clean(dfa, props)
