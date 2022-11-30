import copy
import _pickle as pickle


class FunctionTerm:
    # Represents a term in the end function we want to construct
    # e.g. x^3, 2*f[loc-1],
    def __init__(
        self,
        type="constant",
        c=1,
        exponent1=1,
        exponent2=None,
        index_diff1=None,
        index_diff2=None,
    ):

        self.type = type
        self.coeff = c
        self.exponent1 = exponent1
        self.exponent2 = exponent2
        self.index_diff1 = index_diff1
        self.index_diff2 = index_diff2

        if self.type == "NULL":
            pass
        if self.type == "loc_term":
            assert self.exponent1 is not None
        if self.type == "power_term":
            assert None not in (self.exponent1, self.index_diff1)
        if self.type == "interaction_term":
            assert None not in (
                self.exponent1,
                self.exponent2,
                self.index_diff1,
                self.index_diff2,
            )

    def __str__(self):
        if self.type == 'NULL':
            return 'NULL'
        elif self.type == 'constant':
            return str(self.coeff)+'*1'
        elif self.type == "loc_term":
            return f"{self.coeff}*n^{self.exponent1}"
        elif self.type == "power_term":
            return f"{self.coeff}*f[n-{self.index_diff1}]^{self.exponent1}"
        elif self.type == "interaction_term":
            return f"{self.coeff}*f[n-{self.index_diff1}]^{self.exponent1}*f[n-{self.index_diff2}]^{self.exponent2}"

    def __repr__(self):
        return str(self)

    def updateCoeff(self, c):
        self.coeff = c

    def evaluate(self, f, loc):
        if self.type == "constant":
            return self.coeff
        elif self.type == "loc_term":
            return self.coeff * (loc) ** self.exponent1
        elif self.type == "power_term":
            if loc - 1 - self.index_diff1 < 0:
                return None
            return self.coeff * f[loc - 1 - self.index_diff1] ** self.exponent1
        elif self.type == 'interaction_term':
            if loc-1-self.index_diff1 < 0 or loc-1-self.index_diff2 < 0:
                return None
            return self.coeff * f[loc-1-self.index_diff1]**self.exponent1 * f[loc-1-self.index_diff2]**self.exponent2
        elif self.type == "NULL":
            return 0


class Function:
    def __init__(self, terms=None, coeff=None):
        # terms is a list of FunctionTerm objects and coeff is the coefficients of the terms
        if coeff is not None:
            assert terms is not None and len(terms) == len(coeff)
        self.terms = dict()
        if terms:
            for i, term in enumerate(terms):
                # use pickle to deepcopy term
                term2 = pickle.loads(pickle.dumps(term))
                if coeff is not None:
                    term2.updateCoeff(coeff[i])  # apply coeff to term
                self.addTerm(term2)

    def __str__(self):
        # workaround for comparing None with int, sorted by key (constants before loc_terms before power_terms before interaction terms)
        return "+".join(
            [
                v.__str__()
                for _, v in sorted(
                    self.terms.items(),
                    key=lambda x: tuple([0 if not t else t for t in x[0]]),
                )
                if v.__str__() != "0"
            ]
        )

    def __repr__(self):
        return str(self)

    def addTerm(self, term: FunctionTerm):
        key = (
            term.exponent1 or 0,
            term.exponent2 or 0,
            term.index_diff1,
            term.index_diff2,
            term.type,
        )
        if key not in self.terms:
            self.terms[key] = term
        else:
            # combine like terms
            oldterm = self.terms[key]
            term.coeff += oldterm.coeff
            self.terms[key] = term

    def removeTerm(self, term):
        # Do we need this?
        # TODO
        pass

    def startIndex(self):  # TODO: apparently this function is NOT working
        # the minimum index at which function expression is valid
        # ex: f[n] = f[n-1] + f[n-2], then this method returns 3, since
        # f[3] = f[1]+f[2] but f[2] cannot be expressed as f[1] + f[0] because f[0] doesn't exist
        index_diff_list = [(v.index_diff1 or 0) for _, v in self.terms.items()] + [
            (v.index_diff2 or 0) for _, v in self.terms.items()
        ]
        if len(index_diff_list) == 0:
            return 1
        return max(index_diff_list) + 1

    def evaluate(self, f, loc):
        term_values = [v.evaluate(f, loc) for _, v in self.terms.items()]
        if None in term_values:
            return None
        return sum(term_values)

    def complexity(self):
        # TODO: Occam's Razor, manually define something like
        # len(self.terms)**2 + sum([t.exponent1+(t.exponent2 or 0) for t in self.terms.values()])
        # sus
        pass
