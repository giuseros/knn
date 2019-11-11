#include "matrix.hpp"

#include <cuda.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

#define BLOCK_SIZE 16
#define BLOCK_SIZE_SORT 128

struct DMatrix
{
	int num_cols;
	int num_rows;
	float *data;
};

DMatrix make_dmatrix(knn::Matrix& M)
{
	DMatrix dM;
	dM.num_cols = M.num_cols();
	dM.num_rows = M.num_rows();

	int total_M_size = M.num_cols() * M.num_rows();
	float *dM_ptr;
	cudaMalloc(&dM_ptr, total_M_size*sizeof(float));
	cudaMemcpy(dM_ptr, M.data(), total_M_size*sizeof(float), cudaMemcpyHostToDevice);
	dM.data = dM_ptr;
	return dM;
}

DMatrix make_dmatrix(int num_cols, int num_rows)
{
	DMatrix dM;
	dM.num_cols = num_cols;
	dM.num_rows = num_rows;
	int total_M_size = num_cols * num_rows;
	float *dM_ptr;
	cudaMalloc(&dM_ptr, total_M_size*sizeof(float));
	dM.data = dM_ptr;
	return dM;
}

knn::Matrix extract_matrix(DMatrix dM)
{
	knn::Matrix M(dM.num_cols, dM.num_rows);
	int total_M_size = dM.num_cols * dM.num_rows;
	cudaMemcpy(M.data(), dM.data, total_M_size*sizeof(float), cudaMemcpyDeviceToHost);
	return M;
}

__device__ DMatrix GetSubMatrix(DMatrix A, int row, int col)
{
   DMatrix Asub;
   Asub.num_cols    = BLOCK_SIZE;
   Asub.num_rows   = BLOCK_SIZE;
   Asub.data = &A.data[A.num_cols * BLOCK_SIZE * row + BLOCK_SIZE * col];
   return Asub;
}

__global__
void distance_matrix(DMatrix A, DMatrix Q, DMatrix M)
{
    float D = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < A.num_rows && col < Q.num_rows)
    {
		for (int e = 0; e < A.num_cols; ++e)
		{
			const float t = A.data[row * A.num_cols + e] -  Q.data[col * A.num_cols + e];
			D += t*t;
		}
		M.data[row * M.num_cols + col] = sqrt(D);
    }
}

__global__
void distance_matrix_shared(DMatrix A, DMatrix Q, DMatrix M)
{
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// Each thread block computes one sub-matrix Csub of C
	DMatrix Msub = GetSubMatrix(M, blockRow, blockCol);

	// Each thread computes one element of Msub
	// by accumulating results into D
	float D = 0;

	// Thread row and column within Msub
	int row = threadIdx.y;
	int col = threadIdx.x;
	wmma::fragment<wmma::matrix_a, 16, 16, 16, float, wmma::col_major> a_frag;
	const int global_row = blockRow * BLOCK_SIZE + row;
	const int global_col = blockCol * BLOCK_SIZE + col;
	const bool outside_scope_M = global_row >= M.num_rows || global_col >= M.num_cols;
	const int num_blocks = (A.num_cols  + BLOCK_SIZE - 1) / BLOCK_SIZE;

#pragma unroll
    for (int m = 0; m < num_blocks; ++m) {

        // Get sub-matrix Asub of A
        DMatrix Asub = GetSubMatrix(A, blockRow, m);
    	const bool outside_scope_A = global_row >= A.num_rows || m*BLOCK_SIZE + col >= A.num_cols;

        // Get sub-matrix Bsub of B
        DMatrix Qsub = GetSubMatrix(Q, blockCol, m);
        const bool outside_scope_Q = global_row >= Q.num_rows || m*BLOCK_SIZE + col >= Q.num_cols;

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Qs[BLOCK_SIZE][BLOCK_SIZE];
        float Al[16];
        float Bl[16];

        // Load into shared mem
        As[row][col] = (outside_scope_A ? 0 : Asub.data[row*A.num_cols + col]);
        Qs[row][col] = (outside_scope_Q ? 0 : Qsub.data[col*Q.num_cols + row]);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();



        // Multiply Asub and Qsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
        {
        	float t = (As[row][e] - Qs[e][col]);
            D += t*t;
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    if (!outside_scope_M)
    {
    	Msub.data[row * M.num_rows + col] = sqrt(D);
    }

}

__global__
void sort_distances(DMatrix in, DMatrix out, int k)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < in.num_rows)
	{
		float *out_ptr = out.data + x*k;
		float *in_ptr = in.data + x*in.num_cols;

		for (int i = 0; i<k; i++)
		{
			float elem_to_insert = in_ptr[i];
			for (int j = 0; j< i; j++)
			{
				if (out_ptr[j] > elem_to_insert)
				{
					float tmp = out_ptr[j];
					out_ptr[j] = elem_to_insert;
					elem_to_insert = tmp;
				}
			}
			out_ptr[i] = elem_to_insert;
		}

		for (int i = k; i<in.num_cols; i++)
		{
			float elem_to_insert = in_ptr[i];
			for (int j = 0; j< k; j++)
			{
				if (out_ptr[j] > elem_to_insert)
				{
					float tmp = out_ptr[j];
					out_ptr[j] = elem_to_insert;
					elem_to_insert = tmp;
				}
			}
		}
	}
}

namespace knn
{

knn::Matrix knn_cuda(knn::Matrix& V, knn::Matrix& Q, int k)
{
	const int num_vectors = V.num_rows();
	const int num_references = Q.num_rows();
	const int num_dimensions = V.num_cols();

	knn::Matrix out(k, num_references);

	DMatrix dV = make_dmatrix(V);
	DMatrix dQ = make_dmatrix(Q);
	DMatrix dTmp = make_dmatrix(num_references, num_vectors);
	DMatrix dOut = make_dmatrix(k, num_references);

	dim3 dimBlockD(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGridD( (dTmp.num_cols + dimBlockD.x - 1)/ dimBlockD.x, (dTmp.num_rows + dimBlockD.y - 1 )/ dimBlockD.y);
	distance_matrix_shared<<<dimGridD, dimBlockD>>>(dV, dQ, dTmp);

	dim3 dimBlockS(BLOCK_SIZE_SORT);
	dim3 dimGridS((dTmp.num_rows + dimBlockS.x - 1)/ dimBlockS.x);
	sort_distances<<<dimGridS,dimBlockS>>>(dTmp, dOut, k);

	return extract_matrix(dOut);
}

} // namespace knn
