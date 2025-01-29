
#include "cudaLib.cuh"

#define MAX_VAL 4294967295U

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

int verifyGPUVector(float* a, float* b, float* c, float scale, int size) {
	int errorCount = 0;
	float relativeTolerance = 1e-6;
	float absoluteTolerance = 1e-7;
	for (int idx = 0; idx < size; ++idx) {
		float expected = scale * a[idx] + b[idx];
		float diff = abs(c[idx] - expected);
		if (diff > absoluteTolerance && diff > relativeTolerance * abs(expected)) {
			std::cout << abs(c[idx] - expected) << '\n';
			++errorCount;
			#ifndef DEBUG_PRINT_DISABLE
				std::cout << "Idx " << idx << " expected " << scale * a[idx] + b[idx] 
					<< " found " << c[idx] << " = " << a[idx] << " + " << b[idx] << "\n";
			#endif
		}
	}
	return errorCount;
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size)
		y[i] = scale * x[i] + y[i];
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	srand(time(0));
	float scale = static_cast<float>(rand() % MAX_VAL);
	float *x, *y, *result, *d_x, *d_y;
	x = (float*)malloc(vectorSize*sizeof(float));
	y = (float*)malloc(vectorSize*sizeof(float));
	result = (float*)malloc(vectorSize*sizeof(float));

	if (x == NULL || y == NULL || result == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	cudaMalloc(&d_x, vectorSize*sizeof(float));
	cudaMalloc(&d_y, vectorSize*sizeof(float));

	for (uint64_t i=0; i < vectorSize; i++) {
		x[i] = static_cast<float>(rand() % MAX_VAL);
		y[i] = static_cast<float>(rand() % MAX_VAL);
	}

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %3.4f\n", scale);
		printf(" x = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", x[i]);
		}
		printf(" ... }\n");
		printf(" y = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", y[i]);
		}
		printf(" ... }\n");
	#endif

	cudaMemcpy(d_x, x, vectorSize*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, vectorSize*sizeof(float), cudaMemcpyHostToDevice);

	dim3 DimGrid(ceil(vectorSize/256.0), 1, 1);
	if (vectorSize%256) DimGrid.x++;
	dim3 DimBlock(256, 1, 1);

	saxpy_gpu<<<DimGrid,DimBlock>>>(d_x, d_y, scale, vectorSize);

	cudaMemcpy(result, d_y, vectorSize*sizeof(float), cudaMemcpyDeviceToHost);

	#ifndef DEBUG_PRINT_DISABLE 
		printf(" y = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", result[i]);
		}
		printf(" ... }\n");
	#endif

	int errorCount = verifyGPUVector(x, y, result, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	// Insert code here
	curandState_t rng;
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= pSumSize) return;
	curand_init(clock64(), threadId, 0, &rng);

	uint64_t hitCount = 0;
	float x, y;

	for (uint64_t idx = 0; idx < sampleSize; ++idx) {
		x = curand_uniform(&rng);
		y = curand_uniform(&rng);
			
		if ( int(x * x + y * y) == 0 ) {
			++ hitCount;
		}
	}

	pSums[threadId] = hitCount;
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= reduceSize) return;

	uint64_t localSum = 0;
	for (uint64_t i = threadId; i < pSumSize; i += reduceSize) {
		localSum += pSums[i];
	}

	totals[threadId] = localSum;
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	uint64_t *totals, *d_pSums, *d_totals;
	totals = (uint64_t*)malloc(sampleSize*sizeof(uint64_t));

	if (totals == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	cudaMalloc(&d_pSums, sampleSize*sizeof(uint64_t));
	cudaMalloc(&d_totals, sampleSize*sizeof(uint64_t));

	dim3 DimGrid(ceil(sampleSize/256.0), 1, 1);
	if (sampleSize%256) DimGrid.x++;
	dim3 DimBlock(256, 1, 1);

	generatePoints<<<DimGrid,DimBlock>>>(d_pSums, generateThreadCount, sampleSize);

	reduceCounts<<<DimGrid,DimBlock>>>(d_pSums, d_totals, generateThreadCount, reduceThreadCount);

	cudaMemcpy(totals, d_totals, sampleSize*sizeof(uint64_t), cudaMemcpyDeviceToHost);

	uint64_t totalHitCount = 0;
    for (uint64_t i = 0; i < reduceThreadCount; ++i) {
        totalHitCount += totals[i];
    }

	cudaFree(d_pSums);
	cudaFree(d_totals);
	free(totals);

	approxPi = ((double)totalHitCount / sampleSize) / generateThreadCount ;
	approxPi = approxPi * 4.0f;
		
	return approxPi;
}
