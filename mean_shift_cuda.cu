#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

// Parámetros del algoritmo Mean Shift
#define HS 8              // Ancho de banda espacial (vecindario)
#define HR 16             // Ancho de banda de color (diferencia en Lab)
#define MAX_ITER 5        // Iteraciones máximas por píxel
#define TOL_COLOR 0.3f    // No se usa actualmente, pero es la tolerancia de convergencia en color
#define TOL_SPATIAL 0.3f  // No se usa actualmente, pero es la tolerancia espacial

// Calcula distancia euclidiana entre dos colores en Lab
__device__ float colorDist(uchar3 a, uchar3 b) {
    float dl = a.x - b.x;
    float da = a.y - b.y;
    float db = a.z - b.z;
    return sqrtf(dl*dl + da*da + db*db);
}

// Kernel CUDA: aplica el filtro Mean Shift a cada píxel
__global__ void meanShiftKernel(uchar3* input, uchar3* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Coordenada x del píxel
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Coordenada y del píxel
    if (x >= cols || y >= rows) return;

    int idx = y * cols + x;
    uchar3 center = input[idx]; // Punto actual

    // Iteración de convergencia del píxel hacia su modo
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        float3 sum = make_float3(0, 0, 0);
        int count = 0;

        // Recorrer vecindario espacial
        for (int dy = -HS; dy <= HS; dy++) {
            for (int dx = -HS; dx <= HS; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
                    int nIdx = ny * cols + nx;
                    uchar3 neighbor = input[nIdx];

                    // Verificar si el vecino está dentro del rango de color HR
                    if (colorDist(center, neighbor) < HR) {
                        sum.x += neighbor.x;
                        sum.y += neighbor.y;
                        sum.z += neighbor.z;
                        count++;
                    }
                }
            }
        }

        // Calcular promedio de los puntos vecinos válidos
        if (count > 0) {
            center.x = (uchar)(sum.x / count);
            center.y = (uchar)(sum.y / count);
            center.z = (uchar)(sum.z / count);
        }
    }

    output[idx] = center; // Escribir el resultado
}

int main() {
    // Leer imagen desde disco
    Mat image = imread("C:/Users/LUIS FERNANDO/Pictures/arte/THL.jpg");
    if (image.empty()) {
        cerr << "No se pudo abrir la imagen" << endl;
        return -1;
    }

    // Redimensionar imagen y convertir a espacio de color Lab
    resize(image, image, Size(256, 256));
    cvtColor(image, image, COLOR_BGR2Lab);

    int rows = image.rows, cols = image.cols;
    size_t imgSize = rows * cols * sizeof(uchar3);

    // Reservar memoria en CPU y GPU
    uchar3* h_input = (uchar3*)image.data;
    uchar3* h_output = (uchar3*)malloc(imgSize);

    uchar3 *d_input, *d_output;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMemcpy(d_input, h_input, imgSize, cudaMemcpyHostToDevice);

    // Definir tamaño de bloque e invocar kernel
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    // Medir tiempo de ejecución con CUDA
    auto start = chrono::high_resolution_clock::now();
    meanShiftKernel<<<grid, block>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();

    // Copiar imagen procesada de la GPU a la CPU
    cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost);

    // Crear imagen de salida, convertir a BGR y mostrar
    Mat result(rows, cols, CV_8UC3, h_output);
    cvtColor(result, result, COLOR_Lab2BGR);

    // Mostrar tiempo de ejecución
    chrono::duration<double, milli> duration = end - start;
    cout << "Tiempo de ejecución (CUDA): " << duration.count() << " ms" << endl;

    imshow("Original", image);
    imshow("Filtrado CUDA", result);
    waitKey(0);

    // Liberar memoria
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    return 0;
}
