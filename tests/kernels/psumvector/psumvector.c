/* 
 * psumvector.c - A parallel program to sum the elements of a vector
 */

//	compile as: gcc -DGOOD -DN=x -DREPEAT=y psumvector.c -lpthread -lrt -o psumv
//			
//	run as: ./psumv <num_threads>


#include <pthread.h>	
#include <stdio.h>
#include <stdlib.h>	// for atoi() etc
#include <time.h>		// for clock_gettime()
#include <errno.h>	// for perror()

#ifndef 	N
#define 	N		100000000 	// LOGN max = 28 => N=2^28 = 256M ==> x 4 = 1 GB space for 4 byte types
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
#define 	MSG 	"# PSumVector: Good   : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif
#ifdef 	BAD_MA
#define 	MSG 	"# PSumVector: Bad-MA : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif
#ifdef	BAD_FS
#define 	MSG 	"# PSumVector: Bad-FS : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif
#ifndef 	STRIDE
#define	STRIDE	127
#endif

void *sum(void *p);
float elapsed_time_msec(struct timespec *, struct timespec *, long *, long *);	

// Global shared variables
int	 	psum[MAXTHREADS];  	// partial sum computed by each thread
int 		vector[N];			// check against malloc
int 		sumtotal=0; 
int 		numthreads;

int main(int argc, char **argv) 
{
	int 			correctsum=0, result = 0;
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
    	for (i=0; i < N; i++) {
    		vector[i] = i % 3; // also test with random() % 3;
    		correctsum += vector[i];
    	}
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
		result += psum[i];
	correctsum = REPEAT * correctsum;  
   	if (result != correctsum || (correctsum != sumtotal)  )
		printf("# Error! : correctsum=%d, result=%d, sumtotal=%d\n", correctsum, result, sumtotal); 
	GET_TIME(t3);
  	comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
  	total_time = elapsed_time_msec(&t0, &t3, &sec, &nsec);
	#ifdef BAD_FS
	if (numthreads == 1) {
		printf("# PSumVector: Good   : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n", \
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
  perfpoint_START(4);
#endif

	for (j=0 ; j < REPEAT; j++) {
		#ifdef GOOD
    		for (i = start; i < end; i++) {
    			n += (start + (i * STRIDE + k))% 2; 
    			s[myid] += vector[i];
    		}
    		#elif defined BAD_MA			
    		len = end - start;
    		for (k=0; k < STRIDE; k++) {	// bad memory access (strided)
    			//for (i = start; i < end; i++) {
    			//if (myid == 0) 
    			//printf("myid=%2d: k=%d: ", myid, k);
    			
    			for (i = 0; i < len/STRIDE; i++) {
    				n = start + (i * STRIDE + k) ;
    				s[myid] += vector[n];
    				//if (myid == 0) printf("n=%d ", n);
    			}
    			//if (myid == 0) printf("\n");
    		}
    		if ((len/STRIDE)*STRIDE != len) {	// do the remainder, if any 
    			//if (myid == 0) 
    			//printf("myid=%d Remainder: ", myid);
    			for (n=start + (len/STRIDE)*STRIDE; n < end; n++) { // linearly
    				s[myid] += vector[n];
    				//if (myid == 0) printf("n=%d ", n);    				
    			}
    			//if (myid == 0) printf("\n");
    		}
    		#elif defined BAD_FS
    		for (i = start; i < end; i++) {
    			n += (start + (i * STRIDE + k))% 2; 
			    psum[myid] += vector[i];	// causes false sharing among threads
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

