package com.prem.algorithms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class Eulerian {

    // Hierholzer's Algorithm or Eulerian Path/Circuit.
    // https://www.youtube.com/watch?v=8MpoO2zA2l4
    // https://github.com/williamfiset/Algorithms/blob/master/src/main/java/com/williamfiset/algorithms/graphtheory/EulerianPathDirectedEdgesAdjacencyList.java
    // In graph theory, an Eulerian trail (or Eulerian path) is a trail in a finite
    // graph that visits every edge exactly once (allowing for revisiting vertices).
    // Similarly, an Eulerian circuit or Eulerian cycle is an Eulerian trail that
    // starts and ends on the same vertex.

    private int n;
    private int edgeCount;
    private int[] in, out;
    private LinkedList<Integer> path;
    private List<List<Integer>> graph;

    public Eulerian(List<List<Integer>> graph) {
        if (graph == null)
            throw new IllegalArgumentException("Graph cannot be null..!");
        this.graph = graph;
        this.n = graph.size();
        path = new LinkedList<>();
    }

    // Returns a list of edgeCount + 1 node ids that give the Eulerian path or
    // null if no path exists or the graph is disconnected.
    public int[] getEulerianPath() {
        setUp();

        // Identify if the graph has Eulerian path (meaning: if any graph node has
        // inDegree and outDegree difference not equal to one to identify the
        // start and end node)
        if (!graphHasEulerianPath())
            return null;

        dfs(findStartNode());

        // Make sure all edges of the graph were traversed. It could be the
        // case that the graph is disconnected in which case return null.
        if (path.size() != edgeCount + 1)
            return null;

        int[] result = new int[edgeCount + 1];
        for (int i = 0; !path.isEmpty(); i++) {
            result[i] = path.removeFirst();
        }

        return result;
    }

    private void setUp() {
        in = new int[n];
        out = new int[n];

        edgeCount = 0;

        // Compute in and out node degrees.
        for (int from = 0; from < n; from++) {
            for (int to : graph.get(from)) {
                in[to]++;
                out[from]++;
                edgeCount++;
            }
        }
    }

    private boolean graphHasEulerianPath() {
        if (edgeCount == 0)
            return false;
        int startNodes = 0, endNodes = 0;

        for (int i = 0; i < n; i++) {
            if (out[i] - in[i] > 1 || in[i] - out[i] > 1)
                return false;
            else if (out[i] - in[i] == 1)
                startNodes++;
            else if (in[i] - out[i] == 1)
                endNodes++;
        }

        return (endNodes == 0 && startNodes == 0) || (endNodes == 1 && startNodes == 1);
    }

    private int findStartNode() {
        int start = 0;
        for (int i = 0; i < n; i++) {
            // Unique starting node.
            if (out[i] - in[i] == 1)
                return i;
            // Start at a node with an outgoing edge.
            if (out[i] > 0)
                start = i;
        }
        return start;
    }

    // Perform DFS to find Eulerian path.
    private void dfs(int current) {
        while (out[current] != 0) {
            int next = graph.get(current).get(--out[current]);
            dfs(next);
        }
        path.addFirst(current); // stacking
    }

}
