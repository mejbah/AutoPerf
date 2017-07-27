#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/timer.hpp>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include "perfpoint.h"
//#include <stringstream>

using namespace std;

enum { BufferSize = 1<<24};
int  SLsPerCacheLine = 1;

int ibuffer[BufferSize];

using boost::detail::spinlock;
size_t nslp = 41;
spinlock* pslp = 0;

spinlock& getSpinlock(size_t h)
{
  return pslp[ (h%nslp) * SLsPerCacheLine ];
}


void threadFunc(int offset)
{
#ifdef PERFPOINT
	perfpoint_START(1);
#endif

  const size_t mask = BufferSize-1;
  for (size_t ii=0, index=(offset&mask); ii<BufferSize; ++ii, index=((index+1)&mask))
  {
    spinlock& sl = getSpinlock(index);
    sl.lock();
    ibuffer[index] += 1;
    sl.unlock();
  }

#ifdef PERFPOINT
	perfpoint_END();
#endif
};


int main(int argc, char* argv[])
{
  if ( argc>1 )
  {
	stringstream ss;
	ss << argv[1];
	ss >> SLsPerCacheLine; 
  }

  cout << "Using pool size: "<< nslp << endl;
  cout << "sizeof(spinlock): "<< sizeof(spinlock) << endl;
  cout << "SLsPerCacheLine: "<< int(SLsPerCacheLine) << endl;
  const size_t num = nslp * SLsPerCacheLine;
  pslp = new spinlock[num ];
  for (size_t ii=0; ii<num ; ii++)
  { memset(pslp+ii,0,sizeof(*pslp)); }

  const size_t nThreads = 4;
  boost::thread* ppThreads[nThreads];
  const int offset[nThreads] = { 17, 101, 229, 1023 };

  boost::timer timer;

  for (size_t ii=0; ii<nThreads; ii++)
  { ppThreads[ii] = new boost::thread(threadFunc, offset[ii]); }

  for (size_t ii=0; ii<nThreads; ii++)
  { ppThreads[ii]->join(); }

  cout << "Elapsed time: " << timer.elapsed() << endl;

  for (size_t ii=0; ii<nThreads; ii++)
  { delete ppThreads[ii]; }

  delete[] pslp;

  return 0;
}
