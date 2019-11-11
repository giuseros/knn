#ifndef KNN_CUDA__H__
#define KNN_CUDA__H__

#include "matrix.hpp"

namespace knn
{
knn::Matrix knn_cuda(knn::Matrix& a, knn::Matrix& b, int k);
}

#endif
