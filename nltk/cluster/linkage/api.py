# Natural Language Toolkit: Linkage Interfaces
#
# Copyright (C) 2001-2013 NLTK Project
# Author: Angelos Katharopoulos <katharas@gmail.com>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT

class LinkageI(object):
    """
    Interface covering operations used by a hierarchical clusterer to merge the vectors
    """
    def init_linkage(self, dist, vectors):
        """
        Initialize the linkage in order to use the get_next_merge method
        """
        raise NotImplementedError()

    def get_next_merge(self):
        """
        Return the pair of clusters to be merged next
        """
        raise NotImplementedError()
