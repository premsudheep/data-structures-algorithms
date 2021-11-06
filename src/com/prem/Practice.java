package com.prem;

import java.util.*;
import java.util.HashMap;
import java.util.Stack;
import java.util.PriorityQueue;

public class Practice {

    public class Node {
        private String label;
        private List<Edge> edges = new ArrayList<>();

        public Node(String label) {
            this.label = label;
        }

        public void addEdge(Node to, int weight) {
            this.edges.add(new Edge(this, to, weight));
        }

        public List<Edge> getEdges() {
            return edges;
        }
    }

    public class Edge {
        Node from;
        Node to;
        int weight;

        public Edge(Node from, Node to, int weight) {
            this.from = from;
            this.to = to;
            this.weight = weight;
        }
    }

    private Map<String, Node> nodes = new HashMap<>();

    public void addNode(String label) {
        nodes.putIfAbsent(label, new Node(label));
    }

    public void addEdge(String from, String to, int weight) {
        Node fromNode = nodes.get(from);
        Node toNode = nodes.get(to);

        if (fromNode == null || toNode == null)
            throw new IllegalArgumentException();

        fromNode.addEdge(toNode, weight);
        toNode.addEdge(fromNode, weight);
    }

    public void print() {
        for (Node node : nodes.values()) {
            List<Edge> edges = node.getEdges();
            if (!edges.isEmpty())
                System.out.println(edges);
        }
    }

    private class NodeEntry {
        public Node node;
        public int priority;

        public NodeEntry(Node node, int priority) {
            this.node = node;
            this.priority = priority;
        }
    }

    public int getShortestDistance(String from, String to) {

        Node fromNode = nodes.get(from);

        PriorityQueue<NodeEntry> queue = new PriorityQueue<>(Comparator.comparingInt(ne -> ne.priority));

        Map<Node, Integer> distances = new HashMap<>();
        for (Node node : nodes.values())
            distances.put(node, Integer.MAX_VALUE);
        distances.replace(fromNode, 0);

        Set<Node> visited = new HashSet<>();

        queue.add(new NodeEntry(fromNode, 0));

        while (!queue.isEmpty()) {
            Node current = queue.remove().node;
            visited.add(current);

            for (Edge edge : current.getEdges()) {
                Node neighNode = edge.to;
                int neighWeight = edge.weight;
                if (visited.contains(neighNode))
                    continue;

                int newDistance = distances.get(current) + neighWeight;
                if (newDistance < distances.get(neighNode)) {
                    distances.remove(neighNode, newDistance);
                    queue.add(new NodeEntry(neighNode, newDistance));
                }
            }
        }

        return distances.get(nodes.get(to));
    }

    public boolean hasCycle() {
        Set<Node> visited = new HashSet<>();
        for (Node node : nodes.values()) {
            if (!visited.contains(node) && hasCycle(node, null, visited))
                return true;
        }
        return false;
    }

    private boolean hasCycle(Node node, Node parent, Set<Node> visited) {
        visited.add(node);
        for (Edge edge : node.getEdges()) {
            if (edge.to == parent)
                continue;

            if (visited.contains(edge.to) || hasCycle(edge.to, node, visited))
                return true;
        }
        return false;
    }

    public Practice getMinimumSpanningTree() {
        // WeightedGraph tree = new WeightedGraph();
        Practice tree = new Practice();

        PriorityQueue<Edge> edges = new PriorityQueue<>(Comparator.comparingInt(e -> e.weight));

        Node startNode = nodes.values().iterator().next();
        edges.addAll(startNode.getEdges());
        tree.addNode(startNode.label);

        while (tree.nodes.size() < nodes.size()) {
            Edge minEdge = edges.remove();
            Node nextNode = minEdge.to;

            if (tree.containsNode(nextNode.label))
                continue;

            tree.addNode(nextNode.label);
            tree.addEdge(minEdge.from.label, nextNode.label, minEdge.weight);

            for (Edge edge : nextNode.getEdges()) {
                if (!tree.containsNode(edge.to.label))
                    edges.add(edge);
            }
        }
        return tree;
    }

    public boolean containsNode(String label) {
        return nodes.containsKey(label);
    }

}
