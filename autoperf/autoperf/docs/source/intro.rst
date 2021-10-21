What is AutoPerf?
=================

AutoPerf is a state-of-the-art tool for the automated detection of performance anomalies and regressions in multithreaded programs. It operates in two key phases:

- **Profiling**

  - Collects detailed information from annotated sections of a program by running it with a representative workload

  - Sections are automatically annotated based on version control diffs and Clang abstract syntax trees

  - Profiles are derived from zero-overhead hardware telemetry, made possible via the performance application programming interface (PAPI) [#papi]_ library

  - Then, a denoising autoencoder [#autoencoder]_ is trained on the recorded nominal performance data

.. image:: ./images/train.png

- **Detection**

  - Leverages the previously trained autoencoder, which learned the standard operating characteristics

  - Any workload with abnormal performance will encounter a higher reconstruction error when passed through the network

  - These occurences are then clearly reported to the user, alongside the most abnormal performance counters

.. image:: ./images/detect.png

More details about the design and implementation of AutoPerf can be found in the NeurIPS'19 publication. [#autoperf]_

💡 Motivation
*************

Programmers spend a *lot* of time fixing bugs. Correctness bugs are easy to identify through unit and regression tests, but performance bugs are more challenging to identify, especially in multithreaded environments.

That's where AutoPerf comes in. By comparing the performance of new code directly against the previous version with zero-overhead hardare telemetry, is is possible to accurately detect and diagnose anomalous behavior.

Hardware performance counters give a great deal of insight into **why** the performance regressed. For example, increased levels of branch mispredictions may indicate a suboptimal memory access pattern. This fact makes AutoPerf significantly more powerful than simple wall clock timestamps, which may indicate a problem exists but will offer no solutions or hints.

On the other hand, some popular full-scope profilers like Valgrind [#valgrind]_ can provide similar levels of insight, but also induce severe overhead. For highly concurrent applications, this leads to issues with the probe effect [#probe]_, wherein the act of measurement changes the underlying system to such a degree that the measurement is no longer representative.

AutoPerf gives you the best of both worlds: near-zero workload latency along with high levels of detail.

🧨 Limitations
**************

- AutoPerf's detection process relies on a deterministic workload that encompasses all major features of the software, and is representative of the way an average user interacts with the program. Thus, stochastic programs cannot be analyzed.
- A solid grasp on modern computer architecture (pipelining, cache coherency, etc.) must be held to understand the nuances of what a hardware performance counter tells you about the system behavior.
- Long-term performance creep (e.g. each version introducing a slight delay) will not be flagged, because AutoPerf will only check against the previous version by default. With that being said, it can be done manually.

.. [#papi] https://icl.utk.edu/papi/

.. [#autoencoder] https://dl.acm.org/doi/10.1145/1390156.1390294

.. [#autoperf] https://proceedings.neurips.cc/paper/2019/file/9d1827dc5f75b9d65d80e25eb862e676-Paper.pdf

.. [#valgrind] https://www.valgrind.org

.. [#probe] https://en.wikipedia.org/wiki/Probe_effect