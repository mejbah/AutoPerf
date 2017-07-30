/* 
 * pdotproduct.c - A parallel program to sum the elements of a vector
 */

//	compile as: gcc -DGOOD -DN=x -DREPEAT=y pdotproduct.c -mcmodel=large -lpthread -lrt -o pdot
//	use -mcmodel=large with gcc when statically allocating large arrays > 2 GB		
//	run as: ./pdot <num_threads>


#include <pthread.h>	
#include <stdio.h>
#include <stdlib.h>	// for atoi() etc
#include <time.h>		// for clock_gettime()
#include <errno.h>	// for perror()
#ifdef PERFPOINT
#include "perfpoint.h"
#endif
#ifndef 	N
#define 	N		10000000 	// LOGN max = 27 => N=2^27 = 128M ==> x 4 = 512MB/vector for 4 byte types
#endif					// can say N=(1LL << LOGN)

#ifndef 	REPEAT
#define 	REPEAT	1
#endif
#define	CACHELINE	64				// size of cacheline is 64 bytes
#define	DATASIZE	4				// int=4 bytes, long long is 8 bytes
#define 	MAXTHREADS 	CACHELINE/DATASIZE	// max # parallel threads to sum (with false sharing)	

#define 	GET_TIME(x);	if (clock_gettime(CLOCK_MONOTONIC, &(x)) < 0) \
						{ perror("clock_gettime( ):"); exit(EXIT_FAILURE); }

#ifdef 	GOOD
#define 	MSG 	"# PDotProd: Good   : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif
#ifdef 	BAD_MA
#define 	MSG 	"# PDotProd: Bad-MA : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif
#ifdef	BAD_FS
#define 	MSG 	"# PDotProd: Bad-FS : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif
#ifndef 	STRIDE
#define	STRIDE	127
#endif

void *sum(void *p);
float elapsed_time_msec(struct timespec *, struct timespec *, long *, long *);	

// Global shared variables
int 		psum[MAXTHREADS];  	// partial sum computed by each thread
int 		v1[N], v2[N]; //, v3[N];		// malloc method is slow, but use 
int	 	sumtotal=0; 
int 		numthreads;


int main(int argc, char **argv) 
{
	int 			correct=0, computed = 0;
	pthread_t 		tid[MAXTHREADS];
	int 			i, myid[MAXTHREADS];
	struct timespec 	t0, t1, t2, t3;
	unsigned long 	sec, nsec;
	float 		comp_time, total_time; 	// in milli seconds

	GET_TIME(t0);
    	if (argc != 2) {
		printf("Usage: %s <numthreads>\n", argv[0]);
		exit(0);
    	}
    	numthreads = atoi(argv[1]);
    	if (numthreads > MAXTHREADS) {
		printf("numthreads > MAXTHREADS (%d)\n", MAXTHREADS);
		exit(0);
    	}
    	//v1 = (int *) malloc(N * sizeof (int));
    	//v2 = (int *) malloc(N * sizeof (int));
    	for (i=0; i < N; i++) {
    		v1[i] = 1; v2[i] = 2; // more expensive options: (i+1) % 3 or random() % 3;
    		//v3[i] = 0;
    		
    	}
    	correct = 2*N;
		GET_TIME(t1);
  		//total_time = elapsed_time_msec(&t0, &t1, &sec, &nsec);
  		//printf("InitTime(ms)=%8.1f: ", total_time);
    	for (i = 0; i < numthreads; i++) {                  
		myid[i] = i; 
		psum[i] = 0;                                 
		pthread_create(&tid[i], NULL, sum, &myid[i]); 
    	}                                                 
    	for (i = 0; i < numthreads; i++)  {                  
		pthread_join(tid[i], NULL);                   
    	}
	GET_TIME(t2);

    	// For checking: add up the partial sums computed by each thread 
	for (i = 0; i < numthreads; i++)                    
		computed += psum[i];    
	correct = REPEAT * correct;  
   	if (computed != correct || (correct != sumtotal)  )
		printf("# Error! : correct=%d, computed=%d, sumtotal=%d\n", correct, computed, sumtotal);
	//printf("R=%d, correct=%lld, computed=%lld, sumtotal=%lld\n", REPEAT, correct, computed, sumtotal);
	//free(v1); free(v2); 
	GET_TIME(t3);
  	comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
  	total_time = elapsed_time_msec(&t0, &t3, &sec, &nsec);
	#ifdef BAD_FS
	if (numthreads == 1) {
		printf("# PDotProd: Good   : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n", \
			N, numthreads, comp_time, 100.0*comp_time/total_time);
		return 0;
	}
	#endif
	printf(MSG, N, numthreads, comp_time, 100.0*comp_time/total_time);
	return 0;
}
//-----------------------------------------------
void *sum(void *p) 
{
    	int 		myid = *((int *)p);            		 
    	int 		start = (myid * (long long)N)/numthreads;		// if N >>, use int64_t	 
    	int 		end = ((myid+1) * (long long)N) /numthreads;	// if N >>, use int64_t  
    	int		i, j, k=myid, n=1, len;
    	int	 	s[MAXTHREADS]; s[myid] = 0;

#ifdef PERFPOINT
  perfpoint_START(1);
#endif
	
	for (j=0 ; j < REPEAT; j++) {
		#ifdef GOOD
    		for (i = start; i < end; i++) {
    			n += (start + (i * STRIDE + k))% 2; 
    			s[myid] += v1[i] * v2[i];
    		}
    		#elif defined BAD_MA			
    		len = end - start;
		//printf("myid=%2d: start=%8d, end=%8d, len=%8d, len/STRIDE=%8d\n", \
		//myid, start, end, len, len/STRIDE);
    		for (k=0; k < STRIDE; k++) {	// bad memory access (strided)
    			//for (i = start; i < end; i++) {
    			//if (myid == 0) 
    			//printf("myid=%2d: k=%d: ", myid, k);
    			
    			for (i = 0; i < len/STRIDE; i++) {
    				n = start + (i * STRIDE + k) ;
    				s[myid] += v1[n] * v2[n];
    				//v3[n]++;
    				//if (myid == 0) printf("n=%d ", n);
    			}
    			//if (myid == 0) printf("\n");
    		}
    		if ((len/STRIDE)*STRIDE != len) {	// do the remainder, if any 
    			//if (myid == 0) 
    			//printf("myid=%d Remainder: ", myid);
    			for (n=start + (len/STRIDE)*STRIDE; n < end; n++) { // linearly
    				s[myid] += v1[n] * v2[n];
    				//v3[n]++;
    				//if (myid == 0) printf("n=%d ", n);    				
    			}
    			//if (myid == 0) printf("\n");
    		}
    		// check
    		/*
    		for (i=start; i < end; i++){
    			if (v3[i] != 1)	printf("Error: myid=%2d: v3[%d]=%d\n", myid, i, v3[i]);
    		}
    		*/
    		#elif defined BAD_FS
    		for (i = start; i < end; i++) {
    			n += (start + (i * STRIDE + k))% 2; 
				psum[myid] += v1[i] * v2[i];	// causes false sharing among threads
    		}
    		#endif                         

    	}
    	#ifdef GOOD
    	psum[myid] = s[myid];
    	#endif           
    	#ifdef BAD_MA
    	psum[myid] = s[myid];
    	#endif 
    	s[myid] = n;          
    	sumtotal += psum[myid];				// ideally should use locks

#ifdef PERFPOINT
  perfpoint_END();
#endif
   	            
}
//-----------------------------------------------
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

