// Matrix Multiplication using OpenCL on Odroid XU4 GPU
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define MATRIX_SIZE 512

// OpenCL kernel for matrix multiplication
const char* kernelSource = "\n" 
"__kernel void matmul(__global float* A, __global float* B, __global float* C, int N) {\n" 
"    int row = get_global_id(1);\n" 
"    int col = get_global_id(0);\n" 
"    float sum = 0.0;\n" 
"    for (int k = 0; k < N; k++) {\n" 
"        sum += A[row * N + k] * B[k * N + col];\n" 
"    }\n" 
"    C[row * N + col] = sum;\n" 
"}\n";

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error during operation '%s': %d\n", operation, err);
        exit(1);
    }
}

int main() {
    // Matrix dimensions
    int N = MATRIX_SIZE;

    // Allocate memory for matrices A, B, and C
    size_t bytes = N * N * sizeof(float);
    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);

    // Initialize matrices A and B with random values
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Get platform and device information
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    checkError(err, "Getting platform");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    checkError(err, "Getting device");

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Creating command queue");

    // Create buffers for matrices A, B, and C
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    checkError(err, "Creating buffer A");

    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    checkError(err, "Creating buffer B");

    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    checkError(err, "Creating buffer C");

    // Copy matrices A and B to device memory
    err = clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, bytes, A, 0, NULL, NULL);
    checkError(err, "Copying A to device");

    err = clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, bytes, B, 0, NULL, NULL);
    checkError(err, "Copying B to device");

    // Create program from kernel source
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    checkError(err, "Creating program");

    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char buffer[10240];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
        fprintf(stderr, "CL Compilation failed:\n%s\n", buffer);
        exit(1);
    }

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "matmul", &err);
    checkError(err, "Creating kernel");

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    checkError(err, "Setting kernel arg 0");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    checkError(err, "Setting kernel arg 1");

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    checkError(err, "Setting kernel arg 2");

    err = clSetKernelArg(kernel, 3, sizeof(int), &N);
    checkError(err, "Setting kernel arg 3");

    // Define the global and local work size
    size_t globalSize[2] = {N, N};
    size_t localSize[2] = {16, 16};

    // Execute the kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    // Read the result back to host memory
    err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, bytes, C, 0, NULL, NULL);
    checkError(err, "Reading result");

    // Verify the result (optional)
    // Print a small portion of the result matrix
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }

    // Cleanup
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(A);
    free(B);
    free(C);

    return 0;
}
//gcc -o matmul matmul.c -lOpenCL
