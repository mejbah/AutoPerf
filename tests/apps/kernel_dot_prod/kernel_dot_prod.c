
#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include<pthread.h>
#include "stddefines.h"


typedef struct{

	pthread_t th;
	int tid;
	int start;
	int num_elems;	
	int *psum;
	int *v1;
	int *v2;
} pdot_args;

#ifdef FALSE_SHARING 
void *pdot_1(void *t_args) {
#ifdef PERFPOINT
	perfpoint_START(1);
#endif
	pdot_args* args = (pdot_args*)t_args;

	int tid = args->tid;
	int start = args->start;
	int end = start + args->num_elems;
	
	int i;
	printf("thread %d start %d, end %d\n", tid, start,end);
	for (i = start; i < end; i++)
		args->psum[tid] += (args->v1[i] * args->v2[i]);

#ifdef PERFPOINT
	perfpoint_END();
#endif
	int x = somethingNotImportant( args->v1, args->v2, start, end);
	//printf("Just printing %d\n",x);

}

#else

void *pdot_1(void *t_args) {

#ifdef PERFPOINT
	perfpoint_START(1);
#endif

	pdot_args* args = (pdot_args*)t_args;

	int tid = args->tid;
	int start = args->start;
	int end = start + args->num_elems;
	
	int mysum = 0;
	int i;
	printf("thread %d start %d, end %d\n", tid, start,end);
	for (i = start; i < end; i++)
		mysum += args->v1[i] * args->v2[i];
	args->psum[tid] = mysum;
#ifdef PERFPOINT
	perfpoint_END();
#endif
	
	int x = somethingNotImportant( args->v1, args->v2, start, end);
	//printf("Just printing %d\n",x);

}
#endif

int somethingNotImportant( int *v1, int *v2, int start, int end){
#ifdef PERFPOINT
	perfpoint_START(2);
#endif

	int mysum = 0;
	int i;
	for (i = start; i < end; i++)
		mysum += v1[i] * v2[i];
#ifdef PERFPOINT
	perfpoint_END();
#endif
	return mysum;
}

int main( int argc, char **argv ){

	int *psum;
	int *v1, *v2;
	int vector_len = atoi(argv[1]);
	int num_threads = atoi(argv[2]);
	printf("size %d\n", vector_len);
	printf("threads %d\n", num_threads);

	psum = (int*)CALLOC(num_threads, sizeof(int));
	v1 = (int*)CALLOC(vector_len, sizeof(int));
	v2 = (int*)CALLOC(vector_len, sizeof(int));

	srand((unsigned)time(NULL));
	int i;
	for(i=0; i<vector_len; i++){
		int num = (rand() % 1000) + 1;
		v1[i] = num;
		v2[i] = num;
	}

	int block_size = vector_len / num_threads;

	pdot_args* thd_args =  (pdot_args*)CALLOC(num_threads, sizeof(pdot_args));

#ifdef FALSE_SHARING
	printf("Possible false sharing\n");
#else
	printf("No false sharing\n");
#endif

	for(i = 0; i < num_threads; i++){

		thd_args[i].tid = i;
		thd_args[i].start = i*block_size;
		thd_args[i].num_elems = block_size;
		thd_args[i].v1 = v1;
		thd_args[i].v2 = v2;
		thd_args[i].psum = psum;

		if(i == num_threads - 1){
			thd_args[i].num_elems = vector_len - ( i*block_size );
		}
	  int res = pthread_create(&thd_args[i].th, NULL, pdot_1, (void*)&thd_args[i]);
		
		if(res != 0){ 
		//	HANDLE_ERROR(res, "PTHREAD_CREATE")
			printf("PTHREAD_CREATE FAILED, ERRNO: %d\n", res);
		}

	}
	
	for(i = 0; i < num_threads; i++){
		pthread_join(thd_args[i].th, NULL);
	}

	free(v1);	
	free(v2);
	free(psum);
	return 0;
}

