# Natural Language Toolkit: Hierarchical Clusterer
# 
# Copyright (C) 2001-2013 NLTK Project
# Author: Angelos Katharopoulos <katharas@gmail.com>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT
from __future__ import print_function, unicode_literals

try:
    import numpy
except:
    pass

from nltk.cluster.util import VectorSpaceClusterer, Dendrogram, cosine_distance
from nltk.compat import python_2_unicode_compatible
from nltk.cluster.linkage.group_average import GroupAverageLinkage

@python_2_unicode_compatible
class HierarchicalClusterer(VectorSpaceClusterer):
    
    def __init__(self, num_clusters=1, linkage=None, distance=None,
                       normalise=True, svd_dimensions=None):
        VectorSpaceClusterer.__init__(self, normalise, svd_dimensions)
        self._distance = distance if distance is not None else cosine_distance
        self._num_clusters = num_clusters
        self._dendrogram = None
        self._centroids = None
        self._linkage = linkage if linkage is not None else GroupAverageLinkage()

    def cluster(self, vectors, assign_clusters=False, trace=False):
        # stores the merge order
        self._dendrogram = Dendrogram(
            [numpy.array(vector, numpy.float64) for vector in vectors])
        return VectorSpaceClusterer.cluster(self, vectors, assign_clusters, trace)
    
    def cluster_vectorspace(self, vectors, trace=False):
        N = len(vectors)
        index_map = range(N)

        # initialize the linkage
        self._linkage.init_linkage(self._distance, vectors)
        
        # we need to perform N-C joins to end up with C clusters
        for _ in range(N-self._num_clusters):
            i,j = self._linkage.get_next_merge()
            if trace: print('merging %d and %d' % (i, j))

            self._dendrogram.merge( index_map[i], index_map[j] )
            for x in range(j+1,N):
                index_map[x] -= 1
            index_map[j] = -1

        self.update_clusters(self._num_clusters)

    def update_clusters(self, num_clusters):
        clusters = self._dendrogram.groups(num_clusters)
        self._centroids = []
        for cluster in clusters:
            assert len(cluster) > 0
            if self._should_normalise:
                centroid = self._normalise(cluster[0])
            else:
                centroid = numpy.array(cluster[0])
            for vector in cluster[1:]:
                if self._should_normalise:
                    centroid += self._normalise(vector)
                else:
                    centroid += vector
            centroid /= float(len(cluster))
            self._centroids.append(centroid)
        self._num_clusters = len(self._centroids)

    def classify_vectorspace(self, vector):
        min_d = self._distance(self._centroids[0],vector)
        min_i = 0
        for i in range(1,self._num_clusters):
            d = self._distance(self._centroids[i],vector)
            if d < min_d:
                min_d = d
                min_i = i
        return min_i

    def dendrogram(self):
        """
        :return: The dendrogram representing the current clustering
        :rtype:  Dendrogram
        """
        return self._dendrogram

    def num_clusters(self):
        return self._num_clusters

    def __repr__(self):
        return '<Hierarchical Clusterer n=%d>' % self._num_clusters

def demo():
    """
    Non-interactive demonstration of the clusterers with simple 2-D data.
    """

    from nltk.cluster import HierarchicalClusterer

    # use a set of tokens with 2D indices
    vectors = [numpy.array(f) for f in [[3, 3], [1, 2], [4, 2], [4, 0], [2, 3], [3, 1]]]

    # test the GAAC clusterer with 4 clusters
    clusterer = HierarchicalClusterer(4)
    clusters = clusterer.cluster(vectors, True)

    print('Clusterer:', clusterer)
    print('Clustered:', vectors)
    print('As:', clusters)
    print()

    # show the dendrogram
    clusterer.dendrogram().show()

    # classify a new vector
    vector = numpy.array([3, 3])
    print('classify(%s):' % vector, end=' ')
    print(clusterer.classify(vector))
    print()


if __name__ == '__main__':
    demo()

