#!/usr/bin/env python3

import networkx as nx

from pgmpy.models import ClusterGraph


class JunctionTree(ClusterGraph):
    """
    Class for representing Junction Tree.

    Junction tree is undirected graph where each node represents a clique
    (list, tuple or set of nodes) and edges represent sepset between two cliques.
    Each sepset in G separates the variables strictly on one side of edge to
    other.

    Parameters
    ----------
    data: input graph
        Data to initialize graph. If data=None (default) an empty graph is
        created. The data is an edge list.

    Examples
    --------
    Create an empty JunctionTree with no nodes and no edges

    >>> from pgmpy.models import JunctionTree
    >>> G = JunctionTree()

    G can be grown by adding clique nodes.

    **Nodes:**

    Add a tuple (or list or set) of nodes as single clique node.

    >>> G.add_node(('a', 'b', 'c'))
    >>> G.add_nodes_from([('a', 'b'), ('a', 'b', 'c')])

    **Edges:**

    G can also be grown by adding edges.

    >>> G.add_edge(('a', 'b', 'c'), ('a', 'b'))

    or a list of edges

    >>> G.add_edges_from([(('a', 'b', 'c'), ('a', 'b')),
    ...                   (('a', 'b', 'c'), ('a', 'c'))])
    """

    def __init__(self, ebunch=None):
        super(JunctionTree, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.clique_beliefs = {}
        self.sepset_beliefs = {}
        self.evidence = {}
        self.is_calibrated = False

    def add_edge(self, u, v, **kwargs):
        """
        Add an edge between two clique nodes.

        Parameters
        ----------
        u, v: nodes
            Nodes can be any list or set or tuple of nodes forming a clique.

        Examples
        --------
        >>> from pgmpy.models import JunctionTree
        >>> G = JunctionTree()
        >>> G.add_nodes_from([('a', 'b', 'c'), ('a', 'b'), ('a', 'c')])
        >>> G.add_edges_from([(('a', 'b', 'c'), ('a', 'b')),
        ...                   (('a', 'b', 'c'), ('a', 'c'))])
        """
        if u in self.nodes() and v in self.nodes() and nx.has_path(self, u, v):
            raise ValueError('Addition of edge between {u} and {v} forms a cycle breaking the '
                             'properties of Junction Tree'.format(u=str(u), v=str(v)))

        super(JunctionTree, self).add_edge(u, v, **kwargs)

    def check_model(self):
        """
        Check the model for various errors. This method checks for the following
        errors. In the same time also updates the cardinalities of all the random
        variables.

        * Checks if clique potentials are defined for all the cliques or not.
        * Check for running intersection property is not done explicitly over
        here as it done in the add_edges method.

        Returns
        -------
        check: boolean
            True if all the checks are passed
        """
        if not nx.is_connected(self):
            raise ValueError('The Junction Tree defined is not fully connected.')

        return super(JunctionTree, self).check_model()

    def copy(self):
        """
        Returns a copy of JunctionTree.

        Returns
        -------
        JunctionTree : copy of JunctionTree

        Examples
        -------
        >>> import numpy as np
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.models import JunctionTree
        >>> G = JunctionTree()
        >>> G.add_edges_from([(('a', 'b', 'c'), ('a', 'b')), (('a', 'b', 'c'), ('a', 'c'))])
        >>> phi1 = DiscreteFactor(['a', 'b'], [1, 2], np.random.rand(2))
        >>> phi2 = DiscreteFactor(['a', 'c'], [1, 2], np.random.rand(2))
        >>> G.add_factors(phi1,phi2)
        >>> modelCopy = G.copy()
        >>> modelCopy.edges()
        [(('a', 'b'), ('a', 'b', 'c')), (('a', 'c'), ('a', 'b', 'c'))]
        >>> G.factors
        [<DiscreteFactor representing phi(a:1, b:2) at 0xb720ee4c>, <DiscreteFactor representing phi(a:1, c:2) at 0xb4e1e06c>]
        >>> modelCopy.factors
        [<DiscreteFactor representing phi(a:1, b:2) at 0xb4bd11ec>, <DiscreteFactor representing phi(a:1, c:2) at 0xb4bd138c>]

        """
        copy = JunctionTree(self.edges())
        copy.add_nodes_from(self.nodes())
        if self.factors:
            factors_copy = [factor.copy() for factor in self.factors]
            copy.add_factors(*factors_copy)
        return copy

    def _is_converged(self, operation):
        """
        Checks whether the calibration has converged or not. At convergence
        the sepset belief would be precisely the sepset marginal.

        Parameters
        ----------
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.
            if operation == marginalize, it checks whether the junction tree is calibrated or not
            else if operation == maximize, it checks whether the juction tree is max calibrated or not

        Formally, at convergence or at calibration this condition would be satisified for

        .. math:: \sum_{C_i - S_{i, j}} \beta_i = \sum_{C_j - S_{i, j}} \beta_j = \mu_{i, j}

        and at max calibration this condition would be satisfied

        .. math:: \max_{C_i - S_{i, j}} \beta_i = \max_{C_j - S_{i, j}} \beta_j = \mu_{i, j}
        """
        # If no clique belief, then the clique tree is not calibrated
        if not self.clique_beliefs:
            return False

        for edge in self.edges():
            sepset = frozenset(edge[0]).intersection(frozenset(edge[1]))
            sepset_key = frozenset(edge)
            if (edge[0] not in self.clique_beliefs or edge[1] not in self.clique_beliefs or
                    sepset_key not in self.sepset_beliefs):
                return False

            marginal_1 = getattr(self.clique_beliefs[edge[0]], operation)(list(frozenset(edge[0]) - sepset),
                                                                          inplace=False)
            marginal_2 = getattr(self.clique_beliefs[edge[1]], operation)(list(frozenset(edge[1]) - sepset),
                                                                          inplace=False)
            if marginal_1 != marginal_2 or marginal_1 != self.sepset_beliefs[sepset_key]:
                return False
        return True

    def _calibrate_junction_tree(self, operation):
        """
        Generalized calibration of junction tree or clique using belief propagation. This method can be used for both
        calibrating as well as max-calibrating.
        Uses Lauritzen-Spiegelhalter algorithm or belief-update message passing.

        Parameters
        ----------
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.

        Reference
        ---------
        Algorithm 10.3 Calibration using belief propagation in clique tree
        Probabilistic Graphical Models: Principles and Techniques
        Daphne Koller and Nir Friedman.
        """
        # Initialize clique beliefs as well as sepset beliefs
        self.clique_beliefs = {clique: self.get_factors(clique)
                               for clique in self.nodes()}
        self.sepset_beliefs = {frozenset(edge): None for edge in self.edges()}

        for clique in self.nodes():
            if not self._is_converged(operation=operation):
                neighbors = self.neighbors(clique)
                # update root's belief using nieighbor clique's beliefs
                # upward pass
                for neighbor_clique in neighbors:
                    self._update_beliefs(neighbor_clique, clique, operation=operation)

                bfs_edges = nx.algorithms.breadth_first_search.bfs_edges(self, clique)
                # update the beliefs of all the nodes starting from the root to leaves using root's belief
                # downward pass
                for edge in bfs_edges:
                    self._update_beliefs(edge[0], edge[1], operation=operation)
            else:
                break

    def add_evidence(self, evidence):
        self.is_calibrated = False
        for e in evidence:
            for factor in self.factors:
                if e in factor.variables:
                    factor.add_evidence([(e, evidence[e])])
                    self.evidence[e] = evidence[e]
                    break
        if not self._is_converged(operation='marginalize'):
            self.calibrate()
        else:
            self.is_calibrated = True

    def calibrate(self):
        """
        Calibration using belief propagation in junction tree or clique tree.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.inference import BeliefPropagation
        >>> G = BayesianModel([('diff', 'grade'), ('intel', 'grade'),
        ...                    ('intel', 'SAT'), ('grade', 'letter')])
        >>> diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
        >>> intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])
        >>> grade_cpd = TabularCPD('grade', 3,
        ...                        [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...                         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
        ...                        evidence=['diff', 'intel'],
        ...                        evidence_card=[2, 3])
        >>> sat_cpd = TabularCPD('SAT', 2,
        ...                      [[0.1, 0.2, 0.7],
        ...                       [0.9, 0.8, 0.3]],
        ...                      evidence=['intel'], evidence_card=[3])
        >>> letter_cpd = TabularCPD('letter', 2,
        ...                         [[0.1, 0.4, 0.8],
        ...                          [0.9, 0.6, 0.2]],
        ...                         evidence=['grade'], evidence_card=[3])
        >>> G.add_cpds(diff_cpd, intel_cpd, grade_cpd, sat_cpd, letter_cpd)
        >>> bp = BeliefPropagation(G)
        >>> bp.calibrate()
        """
        self._calibrate_junction_tree(operation='marginalize')
        self.is_calibrated = True

    def max_calibrate(self):
        """
        Max-calibration of the junction tree using belief propagation.

        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.inference import BeliefPropagation
        >>> G = BayesianModel([('diff', 'grade'), ('intel', 'grade'),
        ...                    ('intel', 'SAT'), ('grade', 'letter')])
        >>> diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
        >>> intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])
        >>> grade_cpd = TabularCPD('grade', 3,
        ...                        [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...                         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
        ...                        evidence=['diff', 'intel'],
        ...                        evidence_card=[2, 3])
        >>> sat_cpd = TabularCPD('SAT', 2,
        ...                      [[0.1, 0.2, 0.7],
        ...                       [0.9, 0.8, 0.3]],
        ...                      evidence=['intel'], evidence_card=[3])
        >>> letter_cpd = TabularCPD('letter', 2,
        ...                         [[0.1, 0.4, 0.8],
        ...                          [0.9, 0.6, 0.2]],
        ...                         evidence=['grade'], evidence_card=[3])
        >>> G.add_cpds(diff_cpd, intel_cpd, grade_cpd, sat_cpd, letter_cpd)
        >>> bp = BeliefPropagation(G)
        >>> bp.max_calibrate()
        """
        self._calibrate_junction_tree(operation='maximize')

    def get_cliques(self):
        """
        Returns cliques used for belief propagation.
        """
        return self.nodes()

    def get_clique_beliefs(self):
        """
        Returns clique beliefs. Should be called after the clique tree (or
        junction tree) is calibrated.
        """
        return self.clique_beliefs

    def get_sepset_beliefs(self):
        """
        Returns sepset beliefs. Should be called after clique tree (or junction
        tree) is calibrated.
        """
        return self.sepset_beliefs

    def _update_beliefs(self, sending_clique, recieving_clique, operation):
        """
        This is belief-update method.

        Parameters
        ----------
        sending_clique: node (as the operation is on junction tree, node should be a tuple)
            Node sending the message
        recieving_clique: node (as the operation is on junction tree, node should be a tuple)
            Node recieving the message
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.

        Takes belief of one clique and uses it to update the belief of the
        neighboring ones.
        """
        sepset = frozenset(sending_clique).intersection(frozenset(recieving_clique))
        sepset_key = frozenset((sending_clique, recieving_clique))

        # \sigma_{i \rightarrow j} = \sum_{C_i - S_{i, j}} \beta_i
        # marginalize the clique over the sepset
        sigma = getattr(self.clique_beliefs[sending_clique], operation)(list(frozenset(sending_clique) - sepset),
                                                                        inplace=False)

        # \beta_j = \beta_j * \frac{\sigma_{i \rightarrow j}}{\mu_{i, j}}
        self.clique_beliefs[recieving_clique] *= (sigma / self.sepset_beliefs[sepset_key]
                                                  if self.sepset_beliefs[sepset_key] else sigma)

        # \mu_{i, j} = \sigma_{i \rightarrow j}
        self.sepset_beliefs[sepset_key] = sigma
