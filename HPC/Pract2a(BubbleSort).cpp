#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}
void bubbleSortSequential(vector<int>& arr) {
    for (int i = 0; i < arr.size() - 1; ++i) {
        bool swapped = false;
        for (int j = 0; j < arr.size() - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}
void bubbleSortParallel(vector<int>& arr) {
    for (int i = 0; i < arr.size() - 1; ++i) {
        bool swapped = false;
        #pragma omp parallel for shared(arr) reduction(|:swapped)
        for (int j = 0; j < arr.size() - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}

void printArray(const vector<int>& arr) {
    for (int num : arr) cout << num << " ";
    cout << endl;
}

int main() {
    int N;
    cout << "Enter array size: ";
    cin >> N;
    vector<int> arr(N), arrCopy(N);

    cout << "Enter elements: ";
    for (int i = 0; i < N; ++i) cin >> arr[i];

    arrCopy = arr;

    auto start = chrono::high_resolution_clock::now();
    bubbleSortSequential(arr);
    auto end = chrono::high_resolution_clock::now();
    cout << "Sequential Sorted: "; printArray(arr);
    cout << "Time: " << chrono::duration<double, milli>(end - start).count() << " ms\n";

    start = chrono::high_resolution_clock::now();
    bubbleSortParallel(arrCopy);
    end = chrono::high_resolution_clock::now();
    cout << "Parallel Sorted: "; printArray(arrCopy);
    cout << "Time: " << chrono::duration<double, milli>(end - start).count() << " ms\n";

    return 0;
}
