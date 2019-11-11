#include <algorithm>
#include <cmath>
#include <iostream>

#include "test/knn_reference.hpp"

namespace knn
{
namespace test
{
	Matrix knn_reference(const Matrix& V, const Matrix& Q, int k)
	{
		const int num_vectors = V.num_rows();
		const int num_references = Q.num_rows();
		const int num_dimensions = V.num_cols();

		Matrix all_distances(num_references, num_vectors);

		for (int i = 0; i < num_vectors; i++)
		{
			for (int j = 0; j < num_references; j++)
			{
				float D(0.0f);
				// Calculate distance between vector i and reference j
				for (int k = 0; k < num_dimensions; k++)
				{
					const float t = (V.at(k, i) - Q.at(k, j));
					D += t*t;
				}
				all_distances.set(j, i, std::sqrt(D));
			}
		}

		Matrix out(k, num_vectors);

		float *in_ptr = all_distances.data();
		float *out_ptr = out.data();

		for (int i = 0; i<num_vectors; i++)
		{
			std::sort(in_ptr, in_ptr + num_references);
			std::copy(in_ptr, in_ptr + k, out_ptr);
			in_ptr += num_references;
			out_ptr+= k;
		}
		return out;
	}
}
}
