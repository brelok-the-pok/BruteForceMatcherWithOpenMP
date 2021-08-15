#include <iostream>
#include <fstream>
#include <omp.h>
#include <thread>


using namespace std;
ofstream out;

void FillMatrix(float** m, int N, int M)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			m[i][j] = (double)rand() / RAND_MAX;
		}
	}
}

void CompareArrays(float** arr1, float** arr2, int count)
{
	bool errorFound = false;
	int errorsCount = 0;
	for (int i = 0; i < count; i++)
	{
		if (arr1[i][0] != arr2[i][0])
		{
			errorFound = true;
			errorsCount++;
		}
	}

	if (errorFound)
	{
		out << "Число ошибок: " << errorsCount << endl;
	}
}

float SquaredDifference(float a, float b)
{
	return a * a - 2 * a * b + b * b;
}


void MatchMatrix(float** m1, float** m2, float** res, int N, int M)
{
	thread th;

	auto worker = [m1, m2, res, N, M](int start, int end)
	{
		for (int t = start; t < end; t++)
		{
			for (int j = 0; j < N; j++)
			{
				double rowSum = 0;

				for (int k = 0; k < M; k++)
				{
					rowSum += SquaredDifference(m1[t][k], m2[j][k]);
				}
				if (rowSum < res[t][1])
				{
					res[t][1] = rowSum;
					res[t][0] = j;
				}
			}
		}
	};

	th = thread(worker, 0, N);
	th.join();
}


void MatchMatrixOpenMP(float** m1, float** m2, float** res, int N, int M)
{
	#pragma omp parallel shared(res)
	{
		#pragma omp for
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				double rowSum = 0;
				#pragma omp parallel shared(rowSum)
				{
					
					#pragma omp for
					for (int k = 0; k < M; k++)
					{
						rowSum += SquaredDifference(m1[i][k], m2[j][k]);
					}

					if (rowSum < res[i][1])
					{
						res[i][1] = rowSum;
						res[i][0] = j;
					}
				}
			}
		}
	}
}



void MatchMatrixThread(float** m1, float** m2, float** res, int N, int M)
{
	thread threads[8];

	auto worker = [m1, m2, res, N, M](int start, int end)
	{
		for (int t = start; t < end; t++)
		{
			for (int j = 0; j < N; j++)
			{
				double rowSum = 0;

				for (int k = 0; k < M; k++)
				{
					rowSum += SquaredDifference(m1[t][k], m2[j][k]);
				}
				if (rowSum < res[t][1])
				{
					res[t][1] = rowSum;
					res[t][0] = j;
				}
			}
		}
	};

	for (int i = 0; i < 8; i++)
	{
		int startInd = (N / 8) * i;
		int buf = (N / 8) * (i + 1);
		int endInd = buf > N ? N : buf;

		threads[i] = thread(worker, startInd, endInd);
	}

	for (int i = 0; i < 8; i++)
	{
		threads[i].join();
	}
}

double TestBruteForceMatch(float** m1, float** m2, float** res, int N, int M)
{
	double startTime;
	double endTime;

	startTime = omp_get_wtime();
	MatchMatrix(m1, m2, res, N, M);
	endTime = omp_get_wtime();

	return endTime - startTime;
}

double TestOpenMPBruteForceMatch(float** m1, float** m2, float** res, int N, int M)
{
	double startTime = 0;
	double endTime = 0;

	startTime = omp_get_wtime();
	MatchMatrixOpenMP(m1, m2, res, N, M);
	endTime = omp_get_wtime();

	return endTime - startTime;
}

double TestThreadBruteForceMatch(float** m1, float** m2, float** res, int N, int M)
{
	double startTime = 0;
	double endTime = 0;

	startTime = omp_get_wtime();
	MatchMatrixThread(m1, m2, res, N, M);
	endTime = omp_get_wtime();

	return endTime - startTime;
}

void testFunc(int threadCount, int testCount, int N, int M)
{
	double startTime;
	double endTime;
	double resTime = 0;
	omp_set_num_threads(threadCount);

	float** matrix1 = new float* [N];
	float** matrix2 = new float* [N];
	float** res1 = new float* [N];
	float** res2 = new float* [N];
	for (int i = 0; i < N; i++)
	{
		matrix1[i] = new float[M];
		matrix2[i] = new float[M];
		res1[i] = new float[2];
		res2[i] = new float[2];
		for (int j = 0; j < M; j++)
		{
			matrix1[i][j] = 0;
			matrix2[i][j] = 0;

			res1[i][0] = 0;
			res1[i][1] = FLT_MAX;
			res2[i][0] = 0;
			res2[i][1] = FLT_MAX;
		}
	}
	FillMatrix(matrix1, N, M);
	FillMatrix(matrix2, N, M);

	for (int i = 0; i < testCount; i++)
	{
		resTime += TestBruteForceMatch(matrix1, matrix2, res1, N, M);
	}
	out << "Последовательное выполнение: " << resTime / testCount << endl;

	resTime = 0;
	for (int i = 0; i < testCount; i++)
	{
		resTime += TestOpenMPBruteForceMatch(matrix1, matrix2, res2, N, M);
	}
	out << "Параллельное выполнение: " << resTime / testCount << endl;

	CompareArrays(res1, res2, N);

	// Удаление
	for (int i = 0; i < N; i++)
	{
		delete[] matrix1[i];
		delete[] matrix2[i];
	}
	delete[] matrix1;
	delete[] matrix2;
	delete[] res1;
	delete[] res2;
}
int main(int argc, char* argv[])
{
	setlocale(LC_ALL, "Russian");
	out.open("D:\\data.txt"); // окрываем файл для записи

	for (int i = 1; i < 17; i *= 2)
	{
		out << "_____________________________________________" << endl << endl;
		out << "Потоков: " << i << endl;
		cout << "Потоков: " << i << endl;
		for (int j = 10; j < 1001; j *= 10)
		{
			out << "_____________________________________________" << endl << endl;
			out << "N: " << j << endl;
			for (int k = 10; k < 10001; k*=10)
			{
				out << "_____________________________________________" << endl << endl;
				out << "M: " << k << endl;
				testFunc(i, 10, j, k);
				out << "_____________________________________________" << endl << endl;
			}
			out << "_____________________________________________" << endl << endl;
		}
		out << "_____________________________________________" << endl << endl;

	}
	cout << 22 << endl;
	out.close();
}
