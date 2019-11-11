#include "test/knn_reference.hpp"
#include "knn/knn_cuda.cuh"
#include "matrix.hpp"


#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <functional>
#include <chrono>

using namespace std;
using namespace knn::test;

void fill_matrix(knn::Matrix &a)
{
    random_device rnd_device;

    // Specify the engine and distribution.
    mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    uniform_int_distribution<int> dist {1, 10};

    auto gen = [&dist, &mersenne_engine](){
                   return float(dist(mersenne_engine));
               };

    const int N = a.num_rows() * a.num_cols();
    generate(a.data(), a.data() + N, gen);
}

int main(int argc , char **argv)
{
	bool do_validation = false;
	int num_vectors = 10;
	int num_references = 10;
	int num_dims = 10;
	int k = 4;
	int num_iterations = 10;

	for (int i = 1; i< argc; i++)
	{
		if (std::string(argv[i]) == "--validate")
		{
			do_validation = true;
		}
		else if (std::string(argv[i]) == "-s")
		{
			if (argc > i+3)
			{
				num_vectors = atoi(argv[i+1]);
				num_references = atoi(argv[i+2]);
				num_dims = atoi(argv[i+3]);
				k = atoi(argv[i+4]);
			}
		}
		else if (std::string(argv[i]) == "-i")
		{
			if (argc > i)
			{
				num_iterations = atoi(argv[i+1]);
			}
		}
	}
	knn::Matrix vectors(num_dims, num_vectors);
	knn::Matrix references(num_dims, num_references);
	fill_matrix(vectors);
	fill_matrix(references);

	if (do_validation)
	{
		auto k_nearest_ref = knn_reference(vectors, references, k);

		std::cout<<"****** vectors matrix *****"<<std::endl;
		print_matrix(vectors);

		std::cout<<"****** references matrix *****"<<std::endl;
		print_matrix(references);

		std::cout<<"****** reference result *****"<<std::endl;
		print_matrix(k_nearest_ref);
	}

	auto k_nearest = knn::knn_cuda(vectors,references, k);

	if (do_validation)
	{
		std::cout<<"****** target result *****"<<std::endl;
		print_matrix(k_nearest);
	}

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < num_iterations; i++)
	{
		knn::knn_cuda(vectors,references, k);
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::cout<<"Executed in "<<std::chrono::duration_cast<std::chrono::milliseconds>(end -start).count() / float(num_iterations)<< " ms " << std::endl;


}
