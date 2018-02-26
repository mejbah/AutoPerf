/* 
 * pmatrixcompare.c - A parallel program to compare two matrices element-wise
 */

//	compile as: gcc -DN=10000 -DGOOD -DCHECK -lpthread -lrt -o <prog>
//		
//	run as: ./<prog>  <numthreads>	


#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>		// for clock_gettime()
#include <errno.h>	// for perror()
#include <string.h>	// for memset()

#ifndef 	N
#define 	N 750   	// NxN matrices
#endif
#define	CACHELINE	64				// size of cacheline is 64 bytes
#define	DATASIZE	4				// int = 4 bytes (long long = 8 bytes)
#define 	MAXTHREADS 	CACHELINE/DATASIZE	// max # parallel threads to sum (with false sharing)	
#define 	GET_TIME(x);	if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
						{ perror("clock_gettime( ):"); exit(EXIT_FAILURE); }

#ifdef 	GOOD
#define 	MSG 	"# PMatCompare: Good   : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif
#ifdef 	GOOD2
#define 	MSG 	"# PMatCompare: Good2  : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif
#ifdef 	BAD_MA
#define 	MSG 	"# PMatCompare: Bad-MA : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif
#ifdef	BAD_FS
#define 	MSG 	"# PMatCompare: Bad-FS : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif
#ifndef 	STRIDE
#define	STRIDE	20		// assumes N is a multiple of 20
#endif

// ---------- globals ----------------------
int 		numthreads;   
int 		A[N][N], B[N][N];
int 		pcount[MAXTHREADS];
// ---------- function prototypes ----------------------
void* 	compare(void* slice);
void 		init_matrix(int m[N][N], int v);
void 		print_matrix(int m[N][N]);
float 	elapsed_time_msec(struct timespec *, struct timespec *, long *, long *);

// ---------- main ----------------------
int main(int argc, char* argv[])
{
  	pthread_t 		tid[MAXTHREADS];  
  	int 			i, j, k=0, m, result=0, count=0, myid[MAXTHREADS];
	struct timespec 	t0, t1, t2, t3;
	unsigned long 	sec, nsec;
	float 		comp_time, total_time; 	// in milli seconds

	GET_TIME(t0);
  	if (argc!=2) {
		printf("Usage: %s number_of_threads\n",argv[0]);
    		exit(-1);
  	}
  	numthreads = atoi(argv[1]);
	/*
	for (i=0; i < N; i++) {
		for (j=0; j < N; j++) {
			k = A[i][j] = j % numthreads;
			m = B[i][j] = i % numthreads;
			if (k == m)
				count++;
		}
	}
	*/
	GET_TIME(t1);
  	//total_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
  	//printf("InitTime(ms)=%5.1f: ", total_time);
	for (i = 0; i < numthreads; i++) {
		pcount[i] = 0;
  		myid[i] = i;
    		if (pthread_create (&tid[i], NULL, compare, &myid[i]) != 0 ) {
      		perror("Can't create thread");
      		exit(-1);
    		}
  	}
  	for (i = 0; i < numthreads; i++) 	// main thead waits for other threads to complete
 		pthread_join (tid[i], NULL);
	GET_TIME(t2);
	for (i=0; i < numthreads; i++)
		result += pcount[i];
	//if (result != (REPEAT * count)) printf("# Error! : result(%d) != correct(%d)\n", result, REPEAT * count);
	GET_TIME(t3);
  	comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
  	total_time = elapsed_time_msec(&t0, &t3, &sec, &nsec);
	#ifdef BAD_FS
	if (numthreads == 1) {
		printf("# PMatCompare: Good   : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n", \
			N, numthreads, comp_time, 100.0*comp_time/total_time);
		return 0;
	}
	#endif
	printf(MSG, N, numthreads, comp_time, 100.0*comp_time/total_time);
  	return 0;
}
//-------------------------------------------------
void* compare(void* slice)					// each thread working on its own slice
{
  	int 		myid = *((int *) slice);   		// retrive the slice info
  	int 		from = (myid * N)/numthreads; 	// this works even if N indivisible by numthreads
  	int 		to = ((myid+1) * N)/numthreads; 	
  	int 		i, j, k, mycount[MAXTHREADS]; mycount[myid]=0;
#ifdef PERFPOINT
  perfpoint_START(5);
#endif

  	//printf("computing slice %d (from column %d to %d)\n", myid, from, to-1);
	for (k = 0; k < REPEAT; k++) {
		for (i = from; i < to; i++) {		
			for (j=0; j < N; j++) {
				#ifdef GOOD			// each thread checks slice of rows [from, to]
				//A[i][j] = i % numthreads;
				//B[i][j] = j % numthreads;
				if (A[i][j] == B[i][j])
					mycount[myid]++;
				#elif defined BAD_MA	// each thread checks slice of columns [from, to]
				//A[j][i] = i % numthreads;
				//B[j][i] = j % numthreads;
				if (A[j][i] == B[j][i])
					mycount[myid]++;
				#elif defined BAD_FS	// each thread checks slice of rows [from, to]
				//A[i][j] = i % numthreads;
				//B[i][j] = j % numthreads;
				if (A[i][j] == B[i][j])
					pcount[myid]++;	// but this causes false sharing!
				#endif				
			}
		}
	}
	#ifndef BAD_FS
	pcount[myid] = mycount[myid];
	#endif
#ifdef PERFPOINT
  perfpoint_END();
#endif

}
//------------------------------------------ 	
float elapsed_time_msec(struct timespec *begin, struct timespec *end, long *sec, long *nsec)
{
	if (end->tv_nsec < begin->tv_nsec) {
		*nsec = 1000000000 - (begin->tv_nsec - end->tv_nsec);
    		*sec  = end->tv_sec - begin->tv_sec -1;
    	}
    	else {
		*nsec = end->tv_nsec - begin->tv_nsec;
    		*sec  = end->tv_sec - begin->tv_sec;
	}
	return (float) (*sec) * 1000 + ((float) (*nsec)) / 1000000;

}
//-------------------------------------------------

