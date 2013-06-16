# Natural Language Toolkit: Complete Linkage
# 
# Copyright (C) 2001-2013 NLTK Project
# Author: Angelos Katharopoulos <katharas@gmail.com>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT
from __future__ import print_function, unicode_literals

from nltk.cluster.linkage.heap_linkage import HeapLinkage

class CompleteLinkage(HeapLinkage):
    """
    CompleteLink implements the complete linkage criterion extending the abstract class
    HeapLinkage to be efficient.
    """

    def new_distance(self,xi,xj,i,j):
        """
        Return the maximum distance between the distances of x,i and x,j
        """
        return min(self._dm[xi],self._dm[xj])
