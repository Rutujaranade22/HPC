#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

struct Node {
    int data;
    vector<Node*> neighbors;
};

void parallelDFS(Node* node, vector<bool>& visited) {
    #pragma omp critical
    if (visited[node->data]) return;

    #pragma omp critical
    visited[node->data] = true;

    #pragma omp critical
    cout << node->data << " ";

    #pragma omp parallel for
    for (int i = 0; i < node->neighbors.size(); ++i)
        parallelDFS(node->neighbors[i], visited);
}

int main() {
    vector<Node> graph(5);
    for (int i = 0; i < 5; ++i) graph[i].data = i;

    graph[0].neighbors = {&graph[1], &graph[2]};
    graph[1].neighbors = {&graph[0], &graph[3]};
    graph[2].neighbors = {&graph[0], &graph[4]};
    graph[3].neighbors = {&graph[1]};
    graph[4].neighbors = {&graph[2]};

    vector<bool> visited(5, false);
    cout << "Parallel DFS Output: ";

    #pragma omp parallel
    #pragma omp single
    parallelDFS(&graph[0], visited);

    cout << endl;
    return 0;
}
