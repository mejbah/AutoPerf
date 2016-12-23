#ifndef _XRUN_H
#define _XRUN_H

class xrun {

private:

  xrun()
  {
  }

public:

  static xrun& getInstance() {
    //static char buf[sizeof(xrun)];
    static xrun * singleton = new xrun();

    return *singleton;
  }

  /// @brief Initialize the system.
  void initialize()
  {
	installSignalHandler();
	xthread::getInstance().initialize();
	fprintf(stderr, "xrun initialize before xmemory initialize\n");
//  _memory.initialize();
  }

  void finalize (void)
  {
	xthread::getInstance().finalize();
  }
  /// @brief Install a handler for KILL signals.
  void installSignalHandler() {
    struct sigaction siga;

    // Point to the handler function.
    siga.sa_flags = SA_RESTART | SA_NODEFER;

    siga.sa_handler = sigHandler;
    if (sigaction(SIGINT, &siga, NULL) == -1) {
      perror ("installing SIGINT failed\n");
      exit (-1);
		}
#ifdef USING_SIGUSR2
    if (sigaction(SIGUSR2, &siga, NULL) == -1) {
      perror ("installing SIGUSR2 failed\n");
      exit (-1);
		}
#endif
	}

	static void sigHandler(int signum) {
    if(signum == SIGINT) {
			fprintf(stderr, "Recieved SIGINT, Genearting Report\n");
			//report_mutex_conflicts();	
			//report_thread_waits();
      exit(0);
    }
    else if (signum == SIGUSR2) {
      fprintf(stderr, "Recieving SIGUSR2, check false sharing now:\n");
      //xmemory::getInstance().reportFalseSharing();
    }
  }

  /// The memory manager (for both heap and globals).
//  xmemory&     _memory;
};


#endif
