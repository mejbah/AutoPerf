/* 
 * pmatrixmult.c - A parallel program to mulitply two matrices
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
#define 	N 1000   	// NxN matrices
#endif
#ifndef REPEAT
#define REPEAT 10
#endif
#define	CACHELINE	64				// size of cacheline is 64 bytes
#define	DATASIZE	4				// int = 4 bytes (long long = 8 bytes)
#define 	MAXTHREADS 	CACHELINE/DATASIZE	// max # parallel threads to sum (with false sharing)	
#define 	GET_TIME(x);	if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
						{ perror("clock_gettime( ):"); exit(EXIT_FAILURE); }

#ifdef 	GOOD
#define 	MSG 	"# PMatMult: Good   : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif
#ifdef 	GOOD2
#define 	MSG 	"# PMatMult: Good2  : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif
#ifdef 	BAD_MA
#define 	MSG 	"# PMatMult: Bad-MA : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif
#ifdef	BAD_FS
#define 	MSG 	"# PMatMult: Bad-FS : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif
#ifndef 	STRIDE
#define	STRIDE	20		// assumes N is a multiple of 20
#endif

// ---------- globals ----------------------
int numthreads;   
int A[N][N], B[N][N], C[N][N];
#ifdef CHECK
int Cp[N][N];
#endif
#ifndef GOOD
int psum[MAXTHREADS];
#endif
// ---------- function prototypes ----------------------
void* 	multiply(void* slice);
void 		init_matrix(int m[N][N], int v);
void 		print_matrix(int m[N][N]);
float 	elapsed_time_msec(struct timespec *, struct timespec *, long *, long *);

// ---------- main ----------------------
int main(int argc, char* argv[])
{
  pthread_t 		tid[MAXTHREADS];  
  int 			i, j, k, error=0, myid[MAXTHREADS];
	struct timespec 	t0, t1, t2, t3;
  unsigned long 	sec, nsec;
	float 		comp_time, total_time; 	// in milli seconds

	GET_TIME(t0);
  if (argc!=2) {
	printf("Usage: %s number_of_threads\n",argv[0]);
  		exit(-1);
  }
	/* #ifdef BAD_MA
	if ((N % STRIDE) != 0) {
		printf("N(%d) is not a multiple of STRIDE(%d)\n",N, STRIDE);
    		exit(-1);	
	}
	#endif */
  numthreads = atoi(argv[1]);
  init_matrix(A, 1);
  init_matrix(B, 2);
  init_matrix(C, 0);
  	
	GET_TIME(t1);
  	//total_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
  	//printf("InitTime(ms)=%5.1f: ", total_time);
	for (i = 0; i < numthreads; i++) { 	
  		myid[i] = i;
    	if (pthread_create (&tid[i], NULL, multiply, &myid[i]) != 0 ) {
      		perror("Can't create thread");
      		exit(-1);
    	}
  }
  for (i = 0; i < numthreads; i++) 	// main thead waits for other threads to complete
 		pthread_join (tid[i], NULL);

	GET_TIME(t2);
	
	// check result, by computing Cp sequentially and comparing with C
	#ifdef CHECK
	for (i=0; i < N; i++) {
		for (j=0; j < N; j++) {
			Cp[i][j]= 0;
			for (k=0; k < N; k++)
				Cp[i][j] += A[i][k]*B[k][j];
			if (Cp[i][j] != C[i][j])
				error++;
		}
	}
	if (error)
		printf("# Error! : count = %d *************\n", error);
	#endif
	GET_TIME(t3);
  comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
  total_time = elapsed_time_msec(&t0, &t3, &sec, &nsec);
	#ifdef BAD_FS
	if (numthreads == 1) {
		printf("# PMatMult: Good   : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n", \
			N, numthreads, comp_time, 100.0*comp_time/total_time);
		return 0;
	}
	#endif
	printf(MSG, N, numthreads, comp_time, 100.0*comp_time/total_time);
  	return 0;
}
//-------------------------------------------------
void* multiply(void* slice)					// each thread working on its own slice
{
  	int 		myid = *((int *) slice);   		// retrive the slice info
  	int 		from = (myid * N)/numthreads; 	// this works even if N indivisible by numthreads
  	int 		to = ((myid+1) * N)/numthreads; 	
  	int 		i,j,k, n, m, sum[MAXTHREADS];

  for (m = 0; m < REPEAT; m++) {
#ifdef PERFPOINT
  perfpoint_START(2);
#endif
  //printf("computing slice %d (from column %d to %d)\n", myid, from, to-1);
  	for (j = from; j < to; j++) {			// each thread computes a contiguous slice of columns
  	//for (j = myid; j < N; j += numthreads) {	// each thread striding along columns of C
    		for (i = 0; i < N; i++) {
    			#ifdef GOOD
    			sum[myid] =0;
      		for ( k = 0; k < N; k++) {
      			sum[myid] += A[i][k]*B[k][j];
 			    }
 			    C[i][j] = sum[myid];	// can be +=
    			#elif GOOD2
    			sum[myid] =0;
      	  for ( k = 0; k < N/STRIDE; k++) {
				    for (n=0; n < STRIDE; n++) {
      				sum[myid] += A[i][k*STRIDE + n]*B[k*STRIDE + n][j];
 				    }
			    }
			    if ((N/STRIDE)*STRIDE != N) {
				    for (k=(N/STRIDE)*STRIDE; k < N; k++)
					  sum[myid] += A[i][k]*B[k][j];
			    }
 			    C[i][j] = sum[myid];	// can be +=
 			    #elif defined BAD_FS
    			psum[myid] = 0; 
      		for ( k = 0; k < N; k++) {
 				    psum[myid] += A[i][k]*B[k][j]; // causes false sharing in caches
 			    }
 			    C[i][j] = psum[myid];
 			    #elif defined BAD_MA
			    //
    			sum[myid] =0;
			    for (n=0; n < STRIDE; n++) {
      			for ( k = 0; k < N/STRIDE; k++) {
      				sum[myid] += A[i][k*STRIDE + n]*B[k*STRIDE + n][j];
 				    }
			    }
			    if ((N/STRIDE)*STRIDE != N) {
				    for (k=(N/STRIDE)*STRIDE; k < N; k++)
					    sum[myid] += A[i][k]*B[k][j];
			    }
 			    C[i][j] = sum[myid];	// can be +=
 			    #endif
    		}
  	}
#ifdef PERFPOINT
  perfpoint_END();
#endif
}
  	//printf("finished slice %d\n", myid);
}
//-------------------------------------------------
void init_matrix(int m[N][N], int v)
{
  	int i, j;
  	if (v == 0)  
  		memset(m, 0, N*N*sizeof m[0][0]);
  	/*{
  		for (i = 0; i < N; i++)
    			for (j = 0; j < N; j++)
    			 	m[i][j] = 0;
    	} */
      else	{ // values 0, 1, 2,...,v; 
      v = v+1; 
  		for (i = 0; i < N; i++)
    			for (j = 0; j < N; j++)
    			 	m[i][j] = j % v; // random() % v is more expensive
    	}
}
//-------------------------------------------------
void print_matrix(int m[N][N])
{
  	int i, j;
  	for (i = 0; i < N; i++) {
    		printf("\n\t| ");
    		for (j = 0; j < N; j++)
      		printf("%4d ", m[i][j]);
    		printf("|");
  	}
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

