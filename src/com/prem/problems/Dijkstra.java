public class Dijkstra {

    // Edge
    static class Edge {
        char target;
        int weight;

        Edge(char target, int weight) {
            this.target = target;
            this.weight = weight;
        }
    }

    // priority queue sorting helper
    static class Node implements Comparable<Node> {
        char vertex;
        int distance;

        Node(char vertex, int distance) {
            this.vertex = vertex;
            this.distance = distance;
        }

        @Override
        public int compareTo(Node other) {
            return Integer.compare(this.distance, other.distance);
        }
    }

    public static void runDijkstra(Map<Character, List<Edge>> graph, char startVertex) {

        Map<Character, Integer> distances = new HashMap<>();
        Map<Character, Character> previousVertices = new HashMap<>();
        Set<Character> visited = new HashSet<>();

        // Step 1: Initialize all vertices to Infinity, and start vertex to 0;
        for(char vertex: graph.keySet()) {
            distances.put(vertex, Integer.MAX_VALUE);
            previousVertices.put(vertex, '-'); // '-' means undefined/none
        }
        distances.put(startVertex, 0);

        // Priority Queue serves as an 'unvisited' selector for the smallest known distance
        PriorityQueue<Node> pq = new PriorityQueue<>();
        pq.add(new Node(startVertex, 0));

        // Step 2: Main loop (Repeat untill all reachable options are finalized)
        while(!pq.isEmpty()) {
            // Select the unvisited vertex with the smallest known distance
            Node current = pq.poll();
            char u = current.vertex

            // If its already permanently finalized into the visited list, bypass adding to visited list
            if(visited.contains(u)) {
                continue;
            }

            // Move the current vertex to the vistited list
            visited.add(u);

            // Examine its unvisited neighbours
            for(Edge edge: graph.getOrDefault(u, new ArrayList<>())) {
                char v = edge.target;
                int weight = edge.weight;

                if(visited.contains(v)) {
                    continue; // Skip if already exists
                }

                // Calculate distance from the neighbor from the start vertex
                int currentDistanceU = distances.get(u);
                if(currentDistanceU != Integer.MAX_VALUE) {
                    int calculatedDistance = currentDistanceU + weight;

                    // If calculated distance is less than the know distance, update table
                    if(calculatedDistance < distances.get(v)) {
                        distances.put(v, calculatedDistance);
                        previousVertices.put(v, u) // Track previous node to remember path
                        pq.add(new Node(v, calculatedDistance));
                    }
                }
            }
        }
    } 
}