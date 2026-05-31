#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math_constants.h>

extern "C" {

__global__ void add_memory_kernel(
    cuDoubleComplex* matrix, 
    int size, 
    double k_x, double k_y, double k_z,
    uint64_t structural_seed) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < size && y < size && z < size) {
        int idx = z * size * size + y * size + x;
        double phase = k_x * x + k_y * y + k_z * z;
        
        // 데이터 전송(복사) 없이, VRAM 자체에서 시드(시민권)를 통한 형태(위상 궤적) 창발
        uint64_t h = structural_seed;
        h ^= (uint64_t)x * 0x9e3779b97f4a7c15ULL;
        h ^= (uint64_t)y * 0xbf58476d1ce4e5b9ULL;
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
        h = h ^ (h >> 31);
        
        double amplitude = (double)(h % 10000) / 10000.0;

        // wave = amplitude * exp(i * phase)
        double wave_r = amplitude * cos(phase);
        double wave_i = amplitude * sin(phase);

        cuDoubleComplex wave = make_cuDoubleComplex(wave_r, wave_i);
        matrix[idx] = cuCadd(matrix[idx], wave);
    }
}

__global__ void project_2d_layer_kernel(
    const cuDoubleComplex* matrix,
    int size,
    double k_x, double k_y, double k_z,
    double* projection)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size && y < size) {
        double sum_real = 0.0;
        
        for (int z = 0; z < size; ++z) {
            int idx = z * size * size + y * size + x;
            double obs_phase = k_x * x + k_y * y + k_z * z;
            
            // obs_wave = exp(-i * obs_phase)
            double obs_r = cos(-obs_phase);
            double obs_i = sin(-obs_phase);
            cuDoubleComplex obs_wave = make_cuDoubleComplex(obs_r, obs_i);
            
            cuDoubleComplex val = matrix[idx];
            cuDoubleComplex interference = cuCmul(val, obs_wave);
            
            sum_real += cuCreal(interference);
        }
        
        int proj_idx = y * size + x;
        projection[proj_idx] = sum_real / size;
    }
}

__global__ void project_3d_sphere_kernel(
    const cuDoubleComplex* matrix,
    int size,
    double k_x, double k_y, double k_z,
    double* sphere_surface)
{
    int phi_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int theta_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (phi_idx < size && theta_idx < size) {
        double theta = ((double)theta_idx / size) * CUDART_PI;
        double phi = ((double)phi_idx / size) * 2.0 * CUDART_PI;
        
        double r = size / 2.0;
        double cx = size / 2.0;
        double cy = size / 2.0;
        double cz = size / 2.0;
        
        double px = cx + r * sin(theta) * cos(phi);
        double py = cy + r * sin(theta) * sin(phi);
        double pz = cz + r * cos(theta);
        
        int ix = (int)fmod(px, (double)size);
        int iy = (int)fmod(py, (double)size);
        int iz = (int)fmod(pz, (double)size);
        
        if (ix < 0) ix += size;
        if (iy < 0) iy += size;
        if (iz < 0) iz += size;
        
        double obs_phase = k_x * ix + k_y * iy + k_z * iz;
        double obs_r = cos(-obs_phase);
        double obs_i = sin(-obs_phase);
        cuDoubleComplex obs_wave = make_cuDoubleComplex(obs_r, obs_i);
        
        int idx = iz * size * size + iy * size + ix;
        cuDoubleComplex val = matrix[idx];
        cuDoubleComplex interference = cuCmul(val, obs_wave);
        
        int surf_idx = theta_idx * size + phi_idx;
        sphere_surface[surf_idx] = cuCreal(interference);
    }
}

// Host functions to launch kernels
void launch_add_memory_cuda(
    double* matrix_ptr, int size, 
    double k_x, double k_y, double k_z, 
    uint64_t structural_seed)
{
    cuDoubleComplex* d_matrix;
    
    cudaMalloc(&d_matrix, size * size * size * sizeof(cuDoubleComplex));
    cudaMemcpy(d_matrix, matrix_ptr, size * size * size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    
    dim3 block(8, 8, 8);
    dim3 grid((size + block.x - 1) / block.x, 
              (size + block.y - 1) / block.y, 
              (size + block.z - 1) / block.z);
              
    add_memory_kernel<<<grid, block>>>(d_matrix, size, k_x, k_y, k_z, structural_seed);
    
    cudaMemcpy(matrix_ptr, d_matrix, size * size * size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
}

void launch_project_2d_layer_cuda(
    const double* matrix_ptr, int size,
    double k_x, double k_y, double k_z,
    double* projection_ptr)
{
    cuDoubleComplex* d_matrix;
    double* d_proj;
    
    cudaMalloc(&d_matrix, size * size * size * sizeof(cuDoubleComplex));
    cudaMemcpy(d_matrix, matrix_ptr, size * size * size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_proj, size * size * sizeof(double));
    
    dim3 block(16, 16);
    dim3 grid((size + block.x - 1) / block.x, 
              (size + block.y - 1) / block.y);
              
    project_2d_layer_kernel<<<grid, block>>>(d_matrix, size, k_x, k_y, k_z, d_proj);
    
    cudaMemcpy(projection_ptr, d_proj, size * size * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_matrix);
    cudaFree(d_proj);
}

void launch_project_3d_sphere_cuda(
    const double* matrix_ptr, int size,
    double k_x, double k_y, double k_z,
    double* sphere_ptr)
{
    cuDoubleComplex* d_matrix;
    double* d_sphere;
    
    cudaMalloc(&d_matrix, size * size * size * sizeof(cuDoubleComplex));
    cudaMemcpy(d_matrix, matrix_ptr, size * size * size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_sphere, size * size * sizeof(double));
    
    dim3 block(16, 16);
    dim3 grid((size + block.x - 1) / block.x, 
              (size + block.y - 1) / block.y);
              
    project_3d_sphere_kernel<<<grid, block>>>(d_matrix, size, k_x, k_y, k_z, d_sphere);
    
    cudaMemcpy(sphere_ptr, d_sphere, size * size * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_matrix);
    cudaFree(d_sphere);
}

} // extern "C"
