#include <iostream>
#include <omp.h>
#include <climits>
#include <chrono>
using namespace std;
using namespace std::chrono;

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;
    
    int *arr = new int[n];
    cout << "Enter elements: ";
    for (int i = 0; i < n; i++) cin >> arr[i];

    auto start = high_resolution_clock::now();

    int min_value = INT_MAX, max_value = INT_MIN, sum = 0;
    
    #pragma omp parallel for reduction(min: min_value) reduction(max: max_value) reduction(+: sum)
    for (int i = 0; i < n; i++) {
        min_value = min(min_value, arr[i]);
        max_value = max(max_value, arr[i]);
        sum += arr[i];
    }

    cout << "Min: " << min_value << "\nMax: " << max_value << "\nSum: " << sum 
         << "\nAverage: " << (double)sum / n << endl;

    auto stop = high_resolution_clock::now();
    cout << "Time taken: " << duration_cast<milliseconds>(stop - start).count() << " ms" << endl;

    delete[] arr;
    return 0;
}
