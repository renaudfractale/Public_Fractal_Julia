#include "common.h"
#include <stdio.h>  // Pour fprintf et stderr
#include <stdlib.h> // Pour les fonctions standard C comme malloc
#include <stdint.h>
__global__ void Kernel_Picture(ParameterPicture parameter_picture, long *data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    long index = parameter_picture.Get_index_2D(idx, idy, idz);

    if (index >= 0)
    {
        double2 pos_double = parameter_picture.GetPose_double(idx, idy, idz);
        Complex z(pos_double.x, pos_double.y);
        Complex c(pos_double.x, pos_double.y);
        if (parameter_picture.type_fractal == Type_Fractal::Julia)
        {
            c.x = parameter_picture.coef_julia.x;
            c.y = parameter_picture.coef_julia.y;
        }
        long iter = 0;

        while (z.norm() < 2.0 && iter < parameter_picture.iter_max)
        {
            z = z.power(parameter_picture.power_value) + c;
            iter++;
        }

        data[index] = iter;
    }
}

cudaError_t RUN(ParameterPicture parameter_picture, long *datas, int id_cuda)
{
    size_t size = parameter_picture.Get_size_array_2D() * sizeof(long);
    long *dev_datas = 0;
    cudaError_t cudaStatus;

    const dim3 threadsPerBlock(16, 16, 4);
    const dim3 numBlocks((parameter_picture.lenG + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                         (parameter_picture.lenG + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                         (parameter_picture.lenG + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaStatus = cudaSetDevice(id_cuda);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **)&dev_datas, size);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    Kernel_Picture<<<numBlocks, threadsPerBlock>>>(parameter_picture, dev_datas);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Kernel_Picture launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel_Picture!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(datas, dev_datas, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaFree(dev_datas);
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return cudaStatus;
    }

    return cudaSuccess;

Error:
    cudaFree(dev_datas);
    return cudaStatus;
}
