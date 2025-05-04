#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> leftArr(arr.begin() + left, arr.begin() + mid + 1);
    vector<int> rightArr(arr.begin() + mid + 1, arr.begin() + right + 1);
    int i = 0, j = 0, k = left;
    
    while (i < leftArr.size() && j < rightArr.size()) {
        arr[k++] = (leftArr[i] <= rightArr[j]) ? leftArr[i++] : rightArr[j++];
    }
    while (i < leftArr.size()) arr[k++] = leftArr[i++];
    while (j < rightArr.size()) arr[k++] = rightArr[j++];
}

void mergeSortSequential(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortSequential(arr, left, mid);
        mergeSortSequential(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void mergeSortParallel(vector<int>& arr, int left, int right) {
    if (right - left <= 100) {
        mergeSortSequential(arr, left, right);
    } else {
        int mid = left + (right - left) / 2;
        #pragma omp task shared(arr) firstprivate(left, mid)
        mergeSortParallel(arr, left, mid);
        #pragma omp task shared(arr) firstprivate(mid, right)
        mergeSortParallel(arr, mid + 1, right);
        #pragma omp taskwait
        merge(arr, left, mid, right);
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
    for (int& num : arr) cin >> num;
    
    arrCopy = arr;

    auto start = chrono::high_resolution_clock::now();
    mergeSortSequential(arr, 0, N - 1);
    auto end = chrono::high_resolution_clock::now();
    cout << "Sequential Sorted: "; printArray(arr);
    cout << "Time: " << chrono::duration<double, milli>(end - start).count() << " ms\n";

    start = chrono::high_resolution_clock::now();
    mergeSortParallel(arrCopy, 0, N - 1);
    end = chrono::high_resolution_clock::now();
    cout << "Parallel Sorted: "; printArray(arrCopy);
    cout << "Time: " << chrono::duration<double, milli>(end - start).count() << " ms\n";

    return 0;
}
