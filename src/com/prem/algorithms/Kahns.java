package com.prem.algorithms;

import java.util.List;
import java.util.Queue;
import java.util.ArrayDeque;

public class Kahns {

    // Given a an acyclic graph `g` represented as a adjacency list, return a
    // topological ordering on the nodes of the graph.
    // https://www.youtube.com/watch?v=cIBFEhD77b4
    // https://github.com/williamfiset/Algorithms/blob/master/src/main/java/com/williamfiset/algorithms/graphtheory/Kahns.java
    // Kahn's Algorithm -> (detect cycles and remove those nodes that are in cycle
    // and implement topological sort on the rest of the nodes)

    public int[] kahns(List<List<Integer>> g) throws IllegalAccessException {
        int n = g.size();

        // calculate the in-degree of each node.
        int[] inDegree = new int[n];
        for (List<Integer> edges : g) {
            for (int to : edges) {
                inDegree[to]++;
            }
        }

        // q always contains the set nodes with no incoming edges.
        Queue<Integer> q = new ArrayDeque<>();

        // Find all start nodes. (no incoming edges)
        for (int i = 0; i < n; i++) {
            if (inDegree[i] == 0)
                q.offer(i);
        }

        // Add the staring nodes to order array to identify their to edges with 0
        // in-degree and add them to the queue, else decrease the in-degree(because the
        // current one was just visited).
        int index = 0;
        int[] order = new int[n];
        while (!q.isEmpty()) {
            int current = q.poll();
            order[index++] = current;
            for (int to : g.get(current)) {
                inDegree[to]--;
                if (inDegree[to] == 0)
                    q.offer(to);
            }
        }

        if (index != n) {
            throw new IllegalAccessException("Graph is not acyclic! Detected a cycle");
        }
        return order;
    }
}
