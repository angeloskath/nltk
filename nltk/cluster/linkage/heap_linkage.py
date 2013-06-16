# Natural Language Toolkit: Heap Linkage
# 
# Copyright (C) 2001-2013 NLTK Project
# Author: Angelos Katharopoulos <katharas@gmail.com>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT
from __future__ import print_function, unicode_literals
from heapq import heapify, heappush, heappop

try:
    import numpy
except:
    pass

from nltk.cluster.linkage.api import LinkageI

class HeapLinkage(LinkageI):

    def __init__(self):
        pass

    def init_linkage(self,dist,vectors):
        N = len(vectors)
        # _dm holds the distance matrix
        self._dm = numpy.zeros((N,N),dtype=numpy.float)

        # _versions is an indicator matrix that we have changed a value in the heap
        # because we don't know where an entry is in the heap we increment the version
        # of i,j pair and only accept the value from the heap if the version is correct
        self._versions = numpy.ones((N,N),dtype=numpy.int32) # maybe change this to int16?

        self._heap = []

        # In this list we keep track of the clusters that have been merged with another
        # cluster in order to discard the value from the heap
        self._removed = [False]*N

        # compute the distance matrix
        for i in range(N):
            for j in range(i+1,N):
                d = dist(vectors[i], vectors[j])
                self._dm[ i,j ] = d
                # we only need to fill half the
                # matrix since it is symmetric
                # self._dm[ j,i ] = d
                # self._dm[ i,i ] = 0
                self._heap.append((d,i,j,1))
        # make a heap with the distances
        heapify(self._heap)

    def get_next_merge(self):
        # get the next pair for merging
        d,i,j,v = heappop(self._heap)
        while self._removed[i] or self._removed[j] or v!=self._versions[i,j]:
            d,i,j,v = heappop(self._heap)

        # got the pair now update the distances

        # j is merged into i
        self._removed[j] = True

        # i cluster will change so all the distances with i will change and
        # we need to increment their version for the heap
        self._versions[ :, i] += 1
        self._versions[ i, :] += 1

        # we use three loops because we fill only half the matrix so we need to know
        # the order of x,i,j

        # x<i<j
        for x in range(i):
            if self._removed[x]:
                continue
            d = self.new_distance((x,i),(x,j),i,j)
            self._dm[ x,i ] = d
            heappush(self._heap,(d,x,i,self._versions[x,i]))
        # i<x<j
        for x in range(i+1,j):
            if self._removed[x]:
                continue
            d = self.new_distance((i,x),(x,j),i,j)
            self._dm[ i,x ] = d
            heappush(self._heap,(d,i,x,self._versions[i,x]))
        # i<j<x
        for x in range(j+1,len(self._removed)):
            if self._removed[x]:
                continue
            d = self.new_distance((i,x),(j,x),i,j)
            self._dm[ i,x ] = d
            heappush(self._heap,(d,i,x,self._versions[i,x]))

        return i,j
            
    def new_distance(self,xi,xj,i,j):
        """
        Return the new distance of x,i if we were to merge i and j
        """
        raise NotImplementedError()


