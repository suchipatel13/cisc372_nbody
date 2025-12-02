/************************************************************
 * CISC 372 Assignment 4: N-Body CUDA Simulation
 *
 * Student 1: Suchi Patel
 * Student 2: Alyssa Sanchez
 ************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

__global__
void computeAccelKernel(vector3 *dPos, double *dMass, vector3 *dAccels)
{
    int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int gid_y = blockIdx.y * blockDim.y + threadIdx.y;

    // DO NOT TOUCH MEMORY IF OUT-OF-BOUNDS
    if (gid_x >= NUMENTITIES || gid_y >= NUMENTITIES)
        return;

    int idx = gid_x * NUMENTITIES + gid_y;

    if (gid_x == gid_y)
    {
        dAccels[idx][0] = 0.0;
        dAccels[idx][1] = 0.0;
        dAccels[idx][2] = 0.0;
        return;
    }

    double dx = dPos[gid_x][0] - dPos[gid_y][0];
    double dy = dPos[gid_x][1] - dPos[gid_y][1];
    double dz = dPos[gid_x][2] - dPos[gid_y][2];

    double r2 = dx*dx + dy*dy + dz*dz;
    double r = sqrt(r2);

    double amag = -GRAV_CONSTANT * dMass[gid_y] / r2;

    dAccels[idx][0] = amag * dx / r;
    dAccels[idx][1] = amag * dy / r;
    dAccels[idx][2] = amag * dz / r;
}

__global__
void updateKernel(vector3 *dPos, vector3 *dVel, vector3 *dAccels)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= NUMENTITIES) return;

    vector3 total = {0,0,0};

    int base = gid * NUMENTITIES;

    for (int j = 0; j < NUMENTITIES; j++)
    {
        total[0] += dAccels[base + j][0];
        total[1] += dAccels[base + j][1];
        total[2] += dAccels[base + j][2];
    }

    dVel[gid][0] += total[0] * INTERVAL;
    dVel[gid][1] += total[1] * INTERVAL;
    dVel[gid][2] += total[2] * INTERVAL;

    dPos[gid][0] += dVel[gid][0] * INTERVAL;
    dPos[gid][1] += dVel[gid][1] * INTERVAL;
    dPos[gid][2] += dVel[gid][2] * INTERVAL;
}

extern "C" void compute()
{
    vector3 *dPos, *dVel, *dAccels;
    double *dMass;

    cudaMalloc(&dPos, sizeof(vector3) * NUMENTITIES);
    cudaMalloc(&dVel, sizeof(vector3) * NUMENTITIES);
    cudaMalloc(&dMass, sizeof(double) * NUMENTITIES);
    cudaMalloc(&dAccels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);

    cudaMemcpy(dPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(dVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(dMass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks((NUMENTITIES + 15) / 16, (NUMENTITIES + 15) / 16);

    computeAccelKernel<<<blocks, threads>>>(dPos, dMass, dAccels);
    cudaDeviceSynchronize();

    dim3 threads2(256);
    dim3 blocks2((NUMENTITIES + 255) / 256);

    updateKernel<<<blocks2, threads2>>>(dPos, dVel, dAccels);
    cudaDeviceSynchronize();

    cudaMemcpy(hPos, dPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);

    cudaFree(dPos);
    cudaFree(dVel);
    cudaFree(dMass);
    cudaFree(dAccels);
}

