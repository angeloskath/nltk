# Natural Language Toolkit: Group Average Linkage
# 
# Copyright (C) 2001-2013 NLTK Project
# Author: Angelos Katharopoulos <katharas@gmail.com>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT
from __future__ import print_function, unicode_literals

from nltk.cluster.linkage.heap_linkage import HeapLinkage

class GroupAverageLinkage(HeapLinkage):
    """
    CompleteLink implements the complete linkage criterion extending the abstract class
    HeapLinkage to be efficient.
    """

    def init_linkage(self,dist,vectors):
       self._cluster_size = [1]*len(vectors)
       super(GroupAverageLinkage,self).init_linkage(dist,vectors)

    def get_next_merge(self):
        i,j = super(GroupAverageLinkage,self).get_next_merge()
        self._cluster_size[i] += self._cluster_size[j]
        return i,j

    def new_distance(self,xi,xj,i,j):
        """
        Return the average distance between x,i and x,j the average is weighted by the size of
        the groups.
        """
        si = self._cluster_size[i]
        sj = self._cluster_size[j]

        return (self._dm[xi]*si + self._dm[xj]*sj) / (si+sj)
