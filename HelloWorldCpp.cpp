#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <thread>
#include <random>
#include <cmath>
#include <string>
#include <fstream>
#include <algorithm>

using namespace std;

template <typename T>
struct is_vector : std::false_type {};

template <typename T, typename A>
struct is_vector<std::vector<T, A>> : std::true_type {};

template <typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;

template <typename T>
void logTensor(const T& val) {
    if constexpr (is_vector_v<T>) {
        cout << "[ ";
        for (const auto& item : val) {
            logTensor(item);
        }
        cout << "] ";
    }
    else {
        cout << val << " ";
    }
}

vector<vector<float>> generateRandomMatrix(int rows, int cols, int seed) {
    mt19937 rng(seed);
    uniform_real_distribution<float> dist(-10.0f, 10.0f);

    vector<vector<float>> m(rows, vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            m[i][j] = dist(rng);
        }
    }

    return m;
}

vector<float> getVector(vector<vector<float>>& m, bool flag, int index) {
    vector<float> temp;

    if (flag) {
        for (int i = 0; i < m[0].size(); i++) {
            temp.push_back(m[index][i]);
        }
    }
    else {
        for (int i = 0; i < m.size(); i++) {
            temp.push_back(m[i][index]);
        }
    }

    return temp;
}

vector<vector<float>> transpose(vector<vector<float>>& m2) {
    vector<vector<float>> tM2(m2[0].size(), vector<float>(m2.size()));

    for (int i = 0; i < m2.size(); i++) {
        for (int j = 0; j < m2[i].size(); j++) {
            tM2[j][i] = m2[i][j];
        }
    }

    return tM2;
}

float dotProduct(vector<float>::const_iterator v1Begin, vector<float>::const_iterator v1End, vector<float>::const_iterator v2Begin) {
    float local = 0.0f;

    while (v1Begin != v1End) {
        local += (*v1Begin) * (*v2Begin);
        ++v1Begin;
        ++v2Begin;
    }

    return local;
}

void matmul(vector<vector<float>>& m1, vector<vector<float>>& tM2, vector<vector<float>>& res) {
    for (int i = 0; i < res.size(); i++) {
        for (int j = 0; j < res[i].size(); j++) {
            res[i][j] = dotProduct(m1[i].begin(),m1[i].end(),tM2[j].begin());
        }
    }
}

void matmulPar(vector<vector<float>>& m1, vector<vector<float>>& tM2, vector<vector<float>>& res, int col, int row, int width, int height) {
    for (int i = row; i < row + height; i++) {
        for (int j = col; j < col + width; j++) {
            res[i][j] = dotProduct(m1[i].begin(),m1[i].end(),tM2[j].begin());
        }
    }
}

bool areEqual(const vector<vector<float>>& a,const vector<vector<float>>& b,float eps = 1e-3f) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); ++j) {
            if (fabs(a[i][j] - b[i][j]) > eps) {
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char** argv) {
    int M = 512, K = 512, N = 512;

    int numThr = (argc > 1) ? atoi(argv[1]) : static_cast<int>(thread::hardware_concurrency());
    int runs = (argc > 2) ? atoi(argv[2]) : 100;

    cout << "M: " << M << " K: " << K << " N: " << N << '\n';
    cout << "Threads: " << numThr << '\n';
    cout << "Runs: " << runs << '\n';

    vector<long long> seqTimes;
    vector<long long> parTimes;

    ofstream logFile("timings.csv");
    logFile << "run,seq_us,par_us,speedup\n";

    for (int run = 0; run < runs; run++) {
        int seed1 = 42 + run * 2;
        int seed2 = 43 + run * 2;

        vector<vector<float>> m1 = generateRandomMatrix(M, K, seed1);
        vector<vector<float>> m2 = generateRandomMatrix(K, N, seed2);
        vector<vector<float>> res(m1.size(),vector<float>(m2[0].size()));
        vector<vector<float>> resPar(m1.size(),vector<float>(m2[0].size()));
        vector<vector<float>> tM2 = transpose(m2);

        auto startSeq = chrono::high_resolution_clock::now();
        matmul(m1, tM2, res);
        auto endSeq = chrono::high_resolution_clock::now();
        long long seqDuration =chrono::duration_cast<chrono::microseconds>(endSeq - startSeq).count();

        vector<thread> thrs(numThr);

        int gridRow = 1;
        int gridCol = numThr;

        for (int i = 2; i * i < numThr + 1; i++) {
            if (numThr % i == 0) {
                gridRow = i;
                gridCol = numThr / i;
            }
        }

        int stepSizeRow = res.size() / gridRow;
        int resRow = res.size() % gridRow;

        int stepSizeCol = res[0].size() / gridCol;
        int resCol = res[0].size() % gridCol;

        int localH = 0;
        int threadIdx = 0;

        auto startPar = chrono::high_resolution_clock::now();

        for (int i = 0; i < gridRow; i++) {
            int localW = 0;
            int gridHeight = stepSizeRow + (i < resRow ? 1 : 0);

            for (int j = 0; j < gridCol; j++) {
                int gridWidth = stepSizeCol + (j < resCol ? 1 : 0);
                thrs[threadIdx++] = thread(matmulPar, ref(m1), ref(tM2), ref(resPar), localW, localH, gridWidth, gridHeight);
                localW += gridWidth;
            }

            localH += gridHeight;
        }

        for (int i = 0; i < numThr; i++) {
            thrs[i].join();
        }

        auto endPar = chrono::high_resolution_clock::now();

        long long parDuration =chrono::duration_cast<chrono::microseconds>(endPar - startPar).count();

        seqTimes.push_back(seqDuration);
        parTimes.push_back(parDuration);

        double speedup = static_cast<double>(seqDuration) / parDuration;

        logFile << run + 1 << "," << seqDuration << "," << parDuration << "," << speedup << '\n';

        if (!areEqual(res, resPar)) {
            cout << "Result mismatch at run " << run + 1 << '\n';
        }

        cout << "Run " << run + 1 << " | Seq: " << seqDuration << " microsecond " << (double)seqDuration / 1000 << " millisecond" << " | Par: " << parDuration 
            << " microsecond " << (double)parDuration / 1000 << " millisecond" << " | Speedup: " << speedup << "x\n";
    }

    logFile.close();

    double seqMean = accumulate(seqTimes.begin(), seqTimes.end(), 0.0) / seqTimes.size();

    double parMean = accumulate(parTimes.begin(), parTimes.end(), 0.0) / parTimes.size();

    long long seqMin = *min_element(seqTimes.begin(), seqTimes.end());
    long long seqMax = *max_element(seqTimes.begin(), seqTimes.end());

    long long parMin = *min_element(parTimes.begin(), parTimes.end());
    long long parMax = *max_element(parTimes.begin(), parTimes.end());

    cout << "\nSequential\n";
    cout << "Mean: " << seqMean << " us\n";
    cout << "Min : " << seqMin << " us\n";
    cout << "Max : " << seqMax << " us\n";

    cout << "\nParallel\n";
    cout << "Mean: " << parMean << " us\n";
    cout << "Min : " << parMin << " us\n";
    cout << "Max : " << parMax << " us\n";

    cout << "\nMean speedup: " << seqMean / parMean << "x\n";

    return 0;
}