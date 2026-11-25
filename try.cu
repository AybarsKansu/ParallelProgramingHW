#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

#include <opencv2/core.hpp>       // cv::Mat
#include <opencv2/imgcodecs.hpp>  // imread, imwrite
#include <opencv2/imgproc.hpp>    // görüntü işleme
#include <opencv2/highgui.hpp>    // imshow, waitKey

#define kernel_size 31
#define kernel_radius (kernel_size/2)

#define tile_row 32
#define tile_col 32 // output dim of a blocks calculations
#define shared_row (tile_row + 2 * kernel_radius)
#define shared_col (tile_col + 2 * kernel_radius)

#define resize_size 1024

static_assert(kernel_size > 0 && kernel_size % 2 == 1,
              "kernel_size pozitif ve tek olmali");

float kernel[kernel_size * kernel_size];
__constant__ float d_kernel[kernel_size * kernel_size];

void init_gaussian_kernel() {
    // sigma=0: OpenCV sigma'yi kernel boyutuna gore secer.
    cv::Mat g = cv::getGaussianKernel(kernel_size, 0, CV_32F);
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            kernel[y * kernel_size + x] = g.at<float>(y) * g.at<float>(x);
        }
    }
}

void convolve_cpu_rows(const unsigned char* in, unsigned char* out, int w, int h,
                       int row_begin, int row_end) {
    for (int i = row_begin; i < row_end; i++) {
        for (int j = 0; j < w; j++) {
            float acc = 0;
            for (int ky = -kernel_radius; ky<= kernel_radius; ky++) {
                for (int kx = -kernel_radius; kx<=kernel_radius; kx++) {  
                    int y = std::min(std::max(i+ky, 0), h - 1);
                    int x = std::min(std::max(j+kx, 0), w - 1);
                    acc += in[y*w+x] * kernel[(ky + kernel_radius) * kernel_size + (kx+kernel_radius)];
                }
            }
            out[i*w + j] = cv::saturate_cast<unsigned char>(acc);
        }
    }
}

void convolve_cpu(const unsigned char* in, unsigned char* out, int w, int h) {
    convolve_cpu_rows(in, out, w, h, 0, h);
}

void convolve_cpu_multithread(const unsigned char* in, unsigned char* out,
                             int w, int h, int thread_count) {
    // Giris ortak ve salt okunur; her thread farkli cikti satirlarini yazar.
    thread_count = std::max(1, std::min(thread_count, h));
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (int t = 0; t < thread_count; t++) {
        int begin = h * t / thread_count;
        int end = h * (t + 1) / thread_count;
        workers.emplace_back(convolve_cpu_rows, in, out, w, h, begin, end);
    }
    for (auto& worker : workers) worker.join();
}

__global__ void convolve_gpu(const unsigned char* in, unsigned char* out, int w, int h) {
    __shared__ unsigned char mem[shared_row][shared_col];
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = r * w + c;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < shared_row * shared_col; i+=tile_col * tile_row) {
        int ty = i / shared_col; int tx = i % shared_col;
        int gy = int(blockIdx.y * blockDim.y) + ty - kernel_radius;
        int gx = int(blockIdx.x * blockDim.x) + tx - kernel_radius;

        gy = min(max(gy, 0), h - 1);
        gx = min(max(gx, 0), w - 1);

        mem[ty][tx] = in[gy * w + gx];
    }
    __syncthreads();
    if (r >= h || c >= w) return;
    float acc = 0;
    for (int ky = -kernel_radius; ky<= kernel_radius; ky++) {
        for (int kx = -kernel_radius; kx<=kernel_radius; kx++) {  
            int tx = threadIdx.x + kernel_radius + kx; int ty = threadIdx.y + kernel_radius + ky;
            acc += mem[ty][tx] * d_kernel[(ky + kernel_radius) * kernel_size + (kx+kernel_radius)];
        }
    }
    out[idx] = static_cast<unsigned char>(__float2int_rn(acc));
    
}

// Piksel komsularini dogrudan global memory'den okur.
// Gaussian agirliklari iki GPU surumunde de constant memory'dedir.
__global__ void convolve_gpu_global(const unsigned char* in, unsigned char* out, int w, int h) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= h || c >= w) return;

    float acc = 0;
    for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
        for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
            int y = min(max(r + ky, 0), h - 1);
            int x = min(max(c + kx, 0), w - 1);
            acc += in[y * w + x] *
                   d_kernel[(ky + kernel_radius) * kernel_size + kx + kernel_radius];
        }
    }
    out[r * w + c] = static_cast<unsigned char>(__float2int_rn(acc));
}

__host__ void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

float benchmark_gpu(bool use_shared, const unsigned char* in, unsigned char* out,
                    int w, int h, int repeats) {
    dim3 block(tile_col, tile_row);
    dim3 grid((w + tile_col - 1) / tile_col, (h + tile_row - 1) / tile_row);
    auto launch = [&]() {
        if (use_shared)
            convolve_gpu<<<grid, block>>>(in, out, w, h);
        else
            convolve_gpu_global<<<grid, block>>>(in, out, w, h);
    };

    // Ilk calistirma ve GPU isinmasi olcume dahil degil.
    for (int i = 0; i < 5; i++) launch();
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++) launch();
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaEventSynchronize(stop));
    float ms = 0;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    return ms / repeats;
}

int main(int argc, char** argv)
{
    const bool show = argc > 1 && std::string(argv[1]) == "--show";
    constexpr int cpu_repeats = 3;
    constexpr int gpu_repeats = 30;
    using Clock = std::chrono::steady_clock;
    init_gaussian_kernel();
    cv::Mat img = cv::imread("lena.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "lena.jpg okunamadi\n";
        return 1;
    }
    
    cv::resize(img, img, cv::Size(resize_size, resize_size), 0, 0, cv::INTER_LINEAR);
    int h = img.rows, w = img.cols;
    const int cpu_threads = std::min(h, static_cast<int>(
        std::max(1u, std::thread::hardware_concurrency())));
    cv::Mat cpu(h, w, img.type());
    cv::Mat cpu_multi(h, w, img.type());
    cv::Mat gpu_global(h, w, img.type()), gpu_shared(h, w, img.type());
    std::cout << "Goruntu: " << w << "x" << h << ", Gaussian: "
              << kernel_size << "x" << kernel_size << '\n';
    std::cout << "CPU olculuyor (isinma + " << cpu_repeats << " tekrar)..." << std::endl;
    convolve_cpu(img.data, cpu.data, w, h);
    auto cpu_start = Clock::now();
    for (int i = 0; i < cpu_repeats; i++)
        convolve_cpu(img.data, cpu.data, w, h);
    double cpu_ms = std::chrono::duration<double, std::milli>(Clock::now() - cpu_start).count()
                    / cpu_repeats;

    std::cout << "CPU multithread olculuyor (" << cpu_threads << " thread)..." << std::endl;
    convolve_cpu_multithread(img.data, cpu_multi.data, w, h, cpu_threads);
    cpu_start = Clock::now();
    for (int i = 0; i < cpu_repeats; i++)
        convolve_cpu_multithread(img.data, cpu_multi.data, w, h, cpu_threads);
    // Thread olusturma ve join sureleri de bu olcume dahil.
    double cpu_multi_ms = std::chrono::duration<double, std::milli>(Clock::now() - cpu_start).count()
                          / cpu_repeats;

    unsigned char *d_in = nullptr, *d_out = nullptr;
    size_t bytes = img.total() * img.elemSize();

    cudaCheck(cudaMalloc((void**)&d_in, bytes));
    cudaCheck(cudaMalloc((void**)&d_out, bytes));

    cudaCheck(cudaMemcpyToSymbol(d_kernel, kernel, sizeof(kernel)));
    // Transferler ayri olculur; GPU kernel surelerine dahil degil.
    cudaCheck(cudaDeviceSynchronize());
    auto transfer_start = Clock::now();
    cudaCheck(cudaMemcpy(d_in, img.data, bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());
    double h2d_ms = std::chrono::duration<double, std::milli>(Clock::now() - transfer_start).count();

    std::cout << "GPU olculuyor (surum basina 5 isinma + " << gpu_repeats
              << " tekrar)..." << std::endl;
    float global_ms = benchmark_gpu(false, d_in, d_out, w, h, gpu_repeats);
    cudaCheck(cudaMemcpy(gpu_global.data, d_out, bytes, cudaMemcpyDeviceToHost));
    float shared_ms = benchmark_gpu(true, d_in, d_out, w, h, gpu_repeats);
    transfer_start = Clock::now();
    cudaCheck(cudaMemcpy(gpu_shared.data, d_out, bytes, cudaMemcpyDeviceToHost));
    double d2h_ms = std::chrono::duration<double, std::milli>(Clock::now() - transfer_start).count();
    cudaCheck(cudaFree(d_in));
    cudaCheck(cudaFree(d_out));

    std::cout << std::fixed << std::setprecision(3)
              << "\nOrtalama sureler (ms/islem):\n"
              << "CPU (tek thread):       " << cpu_ms << '\n'
              << "CPU (" << cpu_threads << " thread):        " << cpu_multi_ms << '\n'
              << "GPU global memory:      " << global_ms << '\n'
              << "GPU shared memory:      " << shared_ms << '\n'
              << "CPU / GPU global:       " << cpu_ms / global_ms << "x\n"
              << "CPU / GPU shared:       " << cpu_ms / shared_ms << "x\n"
              << "CPU tek / multithread:  " << cpu_ms / cpu_multi_ms << "x\n"
              << "CPU multi / GPU global: " << cpu_multi_ms / global_ms << "x\n"
              << "CPU multi / GPU shared: " << cpu_multi_ms / shared_ms << "x\n"
              << "Global / shared:        " << global_ms / shared_ms << "x\n"
              << "\nAyri transfer olcumleri (birer ornek):\n"
              << "Host -> Device:         " << h2d_ms << " ms\n"
              << "Device -> Host:         " << d2h_ms << " ms\n"
              << "\nMaksimum piksel farki:\n"
              << "CPU tek / multithread:  " << cv::norm(cpu, cpu_multi, cv::NORM_INF) << '\n'
              << "CPU / GPU global:       " << cv::norm(cpu, gpu_global, cv::NORM_INF) << '\n'
              << "CPU / GPU shared:       " << cv::norm(cpu, gpu_shared, cv::NORM_INF) << '\n'
              << "GPU global / shared:    " << cv::norm(gpu_global, gpu_shared, cv::NORM_INF) << '\n'
              << "Fark = max(|A(y,x) - B(y,x)|); 1, 0-255 araliginda bir gri seviye demek.\n"
              << "CPU/GPU kayan nokta yuvarlamasi nedeniyle 1 seviye fark olabilir.\n";

    if (show) {
        // Ekrana sigmasi icin sadece onizlemeleri kucult.
        auto preview = [](const char* title, const cv::Mat& image) {
            cv::Mat small;
            cv::resize(image, small, cv::Size(768, 768));
            cv::imshow(title, small);
        };
        preview("Input", img);
        preview("CPU", cpu);
        preview("CPU multithread", cpu_multi);
        preview("GPU global", gpu_global);
        preview("GPU shared", gpu_shared);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return 0;
}
