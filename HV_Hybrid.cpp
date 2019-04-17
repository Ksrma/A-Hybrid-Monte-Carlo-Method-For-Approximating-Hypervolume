/*
	C++ implementation of the hybrid algorithm to approximate hypervolume
	
	This program is based on our paper[1]. Please refer to it for more details.
	
	It provides a (epsilon, delta)-approximation for computing hypervolume 
	indicator of a set of points. It combines MCM2RV, a variant of the basic 
	Monte Carlo method[2], and the FPRAS algorithm[3]. 
	
	This program can be compiled into an executable file, e.g.:
	
	c++ -O3 HV_Hybrid.cpp -o HV_Hybrid.exe
	
	, or be transplanted into other programs with functions "check_ratio_mc" and "HV_Hybrid".	
	
	Program usage: 
	HV_Hybrid <number of points> <dimension> <eps> <delta> <round> <point set file> <reference point file> <seed>
		- Input:
			number of points n
			dimension m
			eps: desired approximation error bound
			delta: error probability
			round: number of rounds in the hybrid algorithm
			point set file: n rows of m-D vector
			reference point file: m-D vector (assuming dominated by y point in the point set)
			seed: random seed, using time(0) by default
		- Output: 
			hv: approximation result based on Eq. (1) in the paper
			hv2: approximation result based on Eq. (6) in the paper
			(hv and hv2 will be the same as FPRAS if MCM2RV was not used)
			isMC: flag, which algorithm was used by the hybrid algorithm
			      (0 for FPRAS, 1 for MCM2RV)
							 	
	Author: Jingda Deng
	E-mail: jingddeng2-c@my.cityu.edu.hk
	Latest Update: Apr, 2019

	[1] J. Deng, Q. Zhang, “Combining Simple and Adaptive Monte Carlo Methods for 
	Approximating Hypervolume.” submitted to IEEE TEVC (under review)
	[2] R. M. Everson, J. E. Fieldsend, and S. Singh, “Full elite sets for 
	multi-objective optimisation,” in Adaptive Computing in Design and 
	Manufacture V. Springer, 2002, pp. 343–354.
	[3] K. Bringmann and T. Friedrich, “Approximating the volume of unions
	and intersections of high-dimensional geometric objects,” 
	Computational Geometry, vol. 43, no. 6-7, pp. 601–610, 2010.
*/

#include <cmath>
#include <vector>
#include <random>
#include <time.h> 
#include <iostream>
#include <fstream>

using namespace std;

int check_ratio_mc(int n, long long count, long long iter, int round, int frac, double eps, double Minimum, double Vratio, double Tratio)
{	
	double now_eps = sqrt(frac/(double)round)*eps; // eps1 in the paper
	double est_hv = (double)count/n/iter; // V/V_all
	double est_hv1 = est_hv/(1 - now_eps);
	double est_hv2 = est_hv/(1 + now_eps);
	
	double est1 = est_hv1*est_hv1 + max(Vratio - est_hv1, 0.);
	double est2 = est_hv2*est_hv2 + max(Vratio - est_hv2, 0.);
	double est_k1_low;
	
	if(est_hv1 >= 0.5 && est_hv2 <= 0.5)
		est_k1_low = Minimum;
	else
		est_k1_low = min(est1, est2);
	
	double ratio_up = n*max(est1, est2)*Tratio;
	double ratio_low = n*est_k1_low*Tratio;
	
	// use MCM2RV
	if(ratio_up < (double)(frac - round) / (double)frac)
		return 1;

	// use FPRAS
	if(ratio_low > (double)(frac - round) / (double)frac)
		return 0;
	
	// go to next round
	return -1;
}

double HV_Hybrid(double **points, double eps, double delta, int frac, double *ref, int n, int m, long long seed, double &hv2, int &isMC)
{
	int i, j, k;
	double hv = 0.;
	hv2 = 0.;
	isMC = 0;

	double *sample = new double[m];
	double *vol = new double[n];
	double *lower = new double[m];

	// count numbers
	long long T = 8*log(2./delta)*(1+eps)*n/(eps*eps);
	// Here we use n*R(eps, delta) in Eq. (7) and accordingly change other related variables for easier implementation
	long long T2 = 4*log(2./delta)*(1+eps*(1-eps))*n/(eps*eps)/(1-eps)/(1-eps);
	long long count = 0, iter = 0;
	long long trial;

	// volumes
	double V_all = 0.;		// sum of volume of each bounding box for FPRAS
	double V_all2 = 1.;		// sampling space of simple MC
	for(i=0; i<m; i++)
	{
		lower[ i ] = ref[ i ];
	}
	for(i=0; i<n; i++)
	{
		vol[ i ] = 1.0;
		for(j=0; j<m; j++)
		{
			vol[ i ] *= ref[ j ] - points[ i ][ j ];
			if(lower[ j ] > points[ i ][ j ])
				lower[ j ] = points[ i ][ j ];
		}
		V_all += vol[ i ];
	}
	for(i=0; i<m; i++)
		V_all2 *= ref[i] - lower[i];
	for(i=0; i<n; i++)
		vol[ i ] /= V_all;
	for(i=1; i<n; i++)
	{
		vol[ i ] += vol[ i - 1 ];
	}
	
	// some ratios for comparing MCM2RV and FPRAS
	double Tratio = (double)T2/T;
	double Vratio = V_all2/V_all;
	double Minimum = 0.25 + max(0.0, V_all2/V_all - 0.5);

	// random number generators
	mt19937 mt_rand(seed);
	uniform_real_distribution<double> uniform_dis(0.0, 1.0);
	uniform_int_distribution<int> int_dis(0, n-1);

	// other variables
	int id, id2, l, u, better, flag = -1;
	if(sqrt(frac)*eps > 1)
		frac = 1.0/eps/eps/2;

	double r;

	i = 1;
	trial = T*i/frac;
	// divide original FPRAS into frac rounds
	// i is the number of current round
	while(count < trial)
	{
		// choose one point based on the ratio of its volume respect to 
		// the whole volume as in FPRAS
		r = uniform_dis(mt_rand);

		// binary search
		l = 0; u = n - 1; 
		while(l <= u)
		{
			id = (u + l)/2;
			if(r > vol[ id ])
				l = id + 1;
			else
				u = id - 1;
		}

		// sample one point in the box dominated by the l-th point
		for(k=0; k<m; k++)
		{
			sample[ k ] = points[ l ][ k ] + uniform_dis(mt_rand) * (ref[ k ] - points[ l ][ k ]);
		}
		
		while(true)
		{							
			// randomly choose one point in point set, then do dominance comparison
			id2 = int_dis(mt_rand);
			count++;

			k = 0;
			better = 1;
			while( k < m && better )
			{
				better = points[ id2 ][ k ] <= sample[ k ];
				k++;
			}
			
			// this round ends
			if(count >= trial)
			{
				if( better )
					iter++;

				// check the criterion
				flag = check_ratio_mc(n, count, iter, i, frac, eps, Minimum, Vratio, Tratio);

				// if flag<>-1 then break the loop and 
				// use MCM2RV or FPRAS without checking criterion
				if(flag >= 0)
					break;
				else
				{
					// add i, go to next round
					i++;
					trial = T*i/frac;
					if(trial > T) // T is probably not divisible by frac
					{
						trial = T;
						// end of FPRAS, break and output approximation
						if(count >= T)							
							break;
					}					
				}					
				if( better )
					break;
			}
			else
			{
				if( better )
				{
					iter++;
					break;
				}
			}
		}
	}
	// count < T implies flag<>-1
	if(count < T)
	{
		if(flag <= 0)	// use FPRAS without checking criterion
		{
			if(!better)
			{
				// continue the unsuccessful sample in previous loop
				// in order to output the same value as original FPRAS
				while(true)
				{
					if(count>=T)
					{
						hv = (double)T*V_all/n/iter;
						hv2 = hv;
						delete [] lower;
						delete [] sample;
						delete [] vol;
						return hv;
					}

					id2 = int_dis(mt_rand);
					count++;

					k = 0;
					better = 1;
					while( k < m && better )
					{
						better = points[ id2 ][ k ] <= sample[ k ];
						k++;
					}
					if( better )
					{
						iter++;
						break;
					}
				}
			}
			// original FPRAS
			while(count < T)
			{
				r = uniform_dis(mt_rand);

				l = 0; u = n - 1; 
				while(l <= u)
				{
					id = (u + l)/2;
					if(r > vol[ id ])
						l = id + 1;
					else
						u = id - 1;
				}

				for(k=0; k<m; k++)
				{
					sample[ k ] = points[ l ][ k ] + uniform_dis(mt_rand) * (ref[ k ] - points[ l ][ k ]);
				}

				while(true)
				{
					id2 = int_dis(mt_rand);
					count++;

					k = 0;
					better = 1;
					while( k < m && better )
					{
						better = points[ id2 ][ k ] <= sample[ k ];
						k++;
					}

					if(count>=T)
					{
						if( better ) 
							iter++;
						break;
					}
					else
					{
						if( better )
						{
							iter++;
							break;
						}
					}
				}
			}	
			hv = (double)T*V_all/n/iter;
			hv2 = hv;
			delete [] lower;
			delete [] sample;
			delete [] vol;
			return hv;
		}
		else
		{
			// use MCM2RV, reset counters
			long long iter2 = 0;
			long long num_count = 0;
			long long another_count = 0;

			isMC = 1;

			// vector for partially sampling
			vector<uniform_int_distribution<int>> my_int_dis = vector<uniform_int_distribution<int>>(n);
			for(i=0; i<n; i++)
				my_int_dis[i] = uniform_int_distribution<int>(i, n-1);

			while(true)
			{
				// sample in the whole sampling space of MC
				for(k=0; k<m; k++)
				{
					sample[ k ] = lower[ k ] + uniform_dis(mt_rand) * (ref[ k ] - lower[ k ]);
				}

				// find the first dominating point(if any)
				for(i=0; i<n; i++)
				{
					if(another_count + i>=T2)
					{
						hv = V_all2*iter2/(double)(num_count);
						hv2 = (double)V_all*(iter2)/(double)another_count;
						delete [] lower;
						delete [] sample;
						delete [] vol;
						return hv;
					}

					k = 0;
					better = 1;
					while( k < m && better )
					{
						better = points[ i ][ k ] <= sample[ k ];
						k++;
					}
					if( better )
					{
						// add successful counter
						iter2++;
						break;
					}
				}	

				// add total counter
				num_count++;

				if( better )
				{
					// first i points have been checked
					// they do not dominate this sample, skipped
					id2 = my_int_dis[i](mt_rand);
					k = 0;
					better = 1;
					while( k < m && better )
					{
						better = points[ id2 ][ k ] <= sample[ k ];
						k++;
					}
					// add counter S in the paper
					if( better )
						another_count += (n - i);
					if( another_count >= T2)
					{
						// two approximation results based on Eq. (1) and Eq. (6)
						hv = V_all2*iter2/(double)(num_count);
						hv2 = (double)V_all*(iter2)/(double)another_count;
						delete [] sample;
						delete [] lower;
						delete [] vol;
						return hv;
					}
				}
			}
		}
	}
	else
	{
		// end of FPRAS, output the approximation result
		hv = (double)T*V_all/(double)(n*iter);
		hv2 = hv;
		delete [] lower;
		delete [] sample;
		delete [] vol;
		return hv;
	}
}

int main(int argc, char *argv[])
{	
	if (argc < 8)  
	{
		printf("usage: HV_Hybrid <number of points> <dimension> <eps> <delta> <round> <point set file> <reference point file> <seed>(optional) \n");
		return 1;
	}
	int n, m, r;
	double eps, delta;
	sscanf(argv[1], "%d", &n);
	sscanf(argv[2], "%d", &m);
	sscanf(argv[3], "%lf", &eps);
	sscanf(argv[4], "%lf", &delta);
	sscanf(argv[5], "%d", &r);
	long long seed;
	if(argc < 9)
		seed = time(0);
	else
		sscanf(argv[8], "%llu", &seed);
	
	ifstream file1;
	file1.open(argv[6], ios::in);
	if (!file1.good())
	{
		printf("point set file not found \n");
		return 2;
	}
	
	char str[1024];
	double **point = new double*[n];
	for (int i=0; i<n; i++) 
	{
		point[i] = new double[m];
		for (int j=0; j<m; j++) 
		{
			file1 >> str;
			point[i][j] = atof(str);
		}
	}
	file1.close();

	double *ref = new double[m];
	ifstream file2;
	file2.open(argv[7], ios::in);
	if (!file2.good())
	{
		printf("reference point file not found \n");
		return 3;
	}
	for (int i=0; i<m; i++) 
	{
		file2 >> str;
		ref[i] = atof(str);
	}
	
	file2.close();
	
	clock_t t1 = clock();
	
	double hv, hv2;
	int isMC;
	hv = HV_Hybrid(point, eps, delta, r, ref, n, m, seed, hv2, isMC);
	
	clock_t t2 = clock();
	
	printf(" hv1 = %g; hv2 = %g; isMCM2RV = %d\n", hv, hv2, isMC);
	printf(" time: %.3f(s)\n", (t2 - t1) * 1.0 / CLOCKS_PER_SEC);
	
	return 0;
}
