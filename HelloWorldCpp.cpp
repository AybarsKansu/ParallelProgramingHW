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
#include <immintrin.h>

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

// deprecated - for loop
float dotProduct(vector<float>::const_iterator v1Begin, vector<float>::const_iterator v1End, vector<float>::const_iterator v2Begin) {
    float local = 0.0f;

    while (v1Begin != v1End) {
        local += (*v1Begin) * (*v2Begin);
        ++v1Begin;
        ++v2Begin;
    }

    return local;
}

// updated for simd operations
float dotProductOptimized(const float* a, const float* b, size_t size)
{
    __m256 sum = _mm256_setzero_ps();

    size_t i = 0;

    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);

        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    float temp[8];
    _mm256_storeu_ps(temp, sum);

    float result =
        temp[0] + temp[1] + temp[2] + temp[3] +
        temp[4] + temp[5] + temp[6] + temp[7];

    for (; i < size; ++i)
        result += a[i] * b[i];

    return result;
}

void matmul(vector<vector<float>>& m1, vector<vector<float>>& tM2, vector<vector<float>>& res) {
    for (int i = 0; i < res.size(); i++) {
        for (int j = 0; j < res[i].size(); j++) {
            res[i][j] = dotProduct(m1[i].begin(), m1[i].end(), tM2[j].begin());
        }
    }
}

void matmulPar(vector<vector<float>>& m1, vector<vector<float>>& tM2, vector<vector<float>>& res, int col, int row, int width, int height) {
    for (int i = row; i < row + height; i++) {
        for (int j = col; j < col + width; j++) {
            res[i][j] = dotProduct(m1[i].begin(), m1[i].end(), tM2[j].begin());
        }
    }
}

void matmulOptimized(vector<vector<float>>&m1, vector<vector<float>>&tM2, vector<vector<float>>&res) {
    for (int i = 0; i < res.size(); i++) {
        for (int j = 0; j < res[i].size(); j++) {
            res[i][j] = dotProductOptimized(m1[i].data(), tM2[j].data(), m1[i].size());
        }
    }
}

void matmulParOptimized(vector<vector<float>>& m1, vector<vector<float>>& tM2, vector<vector<float>>& res, int col, int row, int width, int height) {
    for (int i = row; i < row + height; i++) {
        for (int j = col; j < col + width; j++) {
            res[i][j] = dotProductOptimized(m1[i].data(), tM2[j].data(), m1[i].size());;
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

struct Tile {
    int col, row, width, height;
};

int main(int argc, char** argv) {
    int M = 512, K = 512, N = 512;
    /*
        Takes numThr, run count and optimized option
    */
    int numThr = (argc > 1) ? atoi(argv[1]) : static_cast<int>(thread::hardware_concurrency());
    int runs = (argc > 2) ? atoi(argv[2]) : 100;
    bool optimized = (argc > 3) ? (string(argv[3]) == "true") : true;

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
        
        long long seqDuration;
        if (!optimized) {
            auto startSeq = chrono::high_resolution_clock::now();
            matmul(m1, tM2, res);
            auto endSeq = chrono::high_resolution_clock::now();
            seqDuration = chrono::duration_cast<chrono::microseconds>(endSeq - startSeq).count();
        }
        else {
            auto startSeq = chrono::high_resolution_clock::now();
            matmulOptimized(m1, tM2, res);
            auto endSeq = chrono::high_resolution_clock::now();
            seqDuration = chrono::duration_cast<chrono::microseconds>(endSeq - startSeq).count();
        }

        // Start of paralel ======================================================
        vector<thread> thrs;
        thrs.reserve(numThr - 1);

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

        // get tiles so parallel don't waste time calculating it
        vector<Tile> tiles;
        int localH = 0;
        for (int i = 0; i < gridRow; i++) {
            int h = stepSizeRow + (i < resRow ? 1 : 0);
            int localW = 0;
            for (int j = 0; j < gridCol; j++) {
                int w = stepSizeCol + (j < resCol ? 1 : 0);
                tiles.push_back({ localW, localH, w, h });
                localW += w;
            }
            localH += h;
        }

        auto worker = optimized ? matmulParOptimized : matmulPar;

        long long parDuration;

        auto startPar = chrono::high_resolution_clock::now();

        for (size_t t = 0; t + 1 < tiles.size(); t++) {
            const Tile& k = tiles[t];
            thrs.emplace_back(worker, ref(m1), ref(tM2), ref(resPar), k.col, k.row, k.width, k.height);
        }

        const Tile& last = tiles.back();
        worker(m1, tM2, resPar, last.col, last.row, last.width, last.height);

        for (auto& t : thrs) t.join();

        auto endPar = chrono::high_resolution_clock::now();
        parDuration = chrono::duration_cast<chrono::microseconds>(endPar - startPar).count();

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