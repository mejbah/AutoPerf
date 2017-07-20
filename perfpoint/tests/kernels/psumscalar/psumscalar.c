/* 
 * psumscalar.c - A simple parallel sum program to sum a series of scalars
 */

//	compile as: gcc -DGOOD -DN=x -DREPEAT=y psumscalar.c -lpthread -lrt -o psums
//			
//	run as: ./psums <num_threads>


#include <pthread.h>	
#include <stdio.h>
#include <stdlib.h>	// for atoi() etc
#include <time.h>		// for clock_gettime()
#include <errno.h>	// for perror()

#ifndef 	N
#define 	N		300000000 	// LOGN max = 28 => N=2^28 = 256M ==> x 4 = 1 GB space for 4 byte types
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
#define 	MSG 	"# PSumScalar: Good   : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif
#ifdef	BAD_FS
#define 	MSG 	"# PSumScalar: Bad-FS : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n"
#endif

void *sum(void *p);
float elapsed_time_msec(struct timespec *, struct timespec *, long *, long *);	

// Global shared variables
int 		psum[MAXTHREADS];  // partial sum computed by each thread
int	 	sumtotal=0; 
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
    
	GET_TIME(t1);
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
	//correctsum =  (long long) REPEAT * ((long long) N/2) *  ((long long)N-1) ;
   	//if ((result != correctsum) || (result != sumtotal)  )
   	if (result != sumtotal )
		printf("# Error! : result=%d, sumtotal=%d, correctsum=%d\n", result, sumtotal, correctsum); 
	GET_TIME(t3);
  	comp_time = elapsed_time_msec(&t1, &t2, &sec, &nsec);
  	total_time = elapsed_time_msec(&t0, &t3, &sec, &nsec);
	#ifdef BAD_FS
	if (numthreads == 1) {
		printf("# PSumScalar: Good   : N=%d : Threads=%d : CompTime(ms)=%.2f : CompTime/TotalTime=%.1f%%\n", \
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
    	long long 	start = (myid * (long long)N)/numthreads;		 
    	long long	end = ((myid+1) * (long long)N) /numthreads; 	
    	int		i, j;
    	int		s[MAXTHREADS]; s[myid] = 0;
#ifdef PERFPOINT
  perfpoint_START(3);
#endif
	for (j=0 ; j < REPEAT; j++) {
    		for (i = start; i < end; i++) {
    			#ifdef GOOD
    			s[myid] += i % 3;
    			#elif defined BAD_FS        
			psum[myid] += i % 3;	// causes false sharing among threads
			#endif                         
    		}
    	}
    	#ifdef GOOD	                                   
    	psum[myid] = s[myid] ;
    	#endif           
    	sumtotal += psum[myid];		// ideally should use locks            
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

