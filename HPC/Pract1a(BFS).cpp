#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

struct Node {
    int data;
    vector<Node*> children;
    Node(int val) : data(val) {}
};

void parallelBFS(Node* root) {
    if (!root) return;
    queue<Node*> q;
    q.push(root);

    while (!q.empty()) {
        int size = q.size();
        vector<Node*> level(size);

        for (int i = 0; i < size; ++i) {
            level[i] = q.front();
            q.pop();
        }

        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            Node* node = level[i];

            #pragma omp critical
            cout << node->data << " ";

            #pragma omp critical
            for (auto child : node->children) q.push(child);
        }
    }
}

int main() {
    Node* root = new Node(1);
    Node* child1 = new Node(2);
    Node* child2 = new Node(3);
    Node* grandchild = new Node(4);

    child2->children.push_back(grandchild);
    root->children = {child1, child2};

    cout << "Parallel BFS Output: ";
    parallelBFS(root);
    cout << endl;

    return 0;
}
