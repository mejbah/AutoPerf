# AutoPerf #

### What is AutoPerf? ###

Autoperf is a tool for automated diagnosis of performance anomalies in multithreaded programs. It operates in two phases:

1. Profiling: Collects hardware performance counters from annotated sections of a program by running it with performance representative inputs.

2. Anomaly Detection: Creates a model of application performance behavior by training an [Autoencoder](http://dl.acm.org/citation.cfm?id=1390294) network. It finds out the best performing network by training for input dataset(collected in profiling phase). AutoPerf uses the trained model for anomaly detection in future executions of the program.

More details about the design and implementatoin of AutoPerf can be found in this conference [paper](https://arxiv.org/abs/1709.07536), which is accepted at [NeurIPS'19](https://nips.cc/Conferences/2019/Schedule?showEvent=14143) for publication.

### Getting Started ###

**Step 1:** Install [Sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html):

`pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints`

**Step 2:** Build the documentation by navigating to `/docs`, and typing `make html`.

**Step 3:** Open the resulting `/docs/build/html/index.html` file in your web browser of choice. This will present you
with installation instructions, example usage guides, and more!