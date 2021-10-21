Examples
========

.. tip:: The examples below can also be viewed `as a video <https://www.youtube.com>`_!

🖥️ Locally
**********

Analyzing software with AutoPerf can be done in two steps: :code:`init` and :code:`detect`.

.. note:: AutoPerf expects to be run inside of a git repository.

1. :code:`init`

To begin using AutoPerf, type the following while inside of your selected baseline:

.. code-block:: console

    $ autoperf init

This will interactively walk you through a series of questions about your software. For example, what kind of build system you're using, a representative workload that can be run from the terminal,
what branch performance should be measured relative to, etc.

This CLI will initialize a :code:`.autoperf` folder at the base level of the repository, containing two files:

- :code:`config.ini` - the configuration file for all AutoPerf behavior
- :code:`COUNTERS` - the list of PAPI events that should be monitored

You can make changes to these files whenever you wish, but the CLI provides a helpful way to instantiate or modify them.

.. tip:: You only have to run **init** once! All further analysis (even on different branches) will use the same information you entered here.

2. :code:`detect`

Detecting performance anomalies with AutoPerf is simple. Change to the branch that holds the potential defect, then run:

.. code-block:: console

    $ autoperf detect {number}

Where :code:`{number}` is an integer representing the number of times the representative workload in executed for **both** training and inference. Higher numbers reduce the false positive rate, but also take longer to execute. You can use `1`, but `5` or even `10` is preferred.

Another factor to consider with runtime is that the workload has to be run separately for each individual performance counter. So, if you record `30` performance counters with `5` runs, the workload will be executed `300` times (:math:`30\times 5\times 2`).

Running the above code snippet will execute the following actions sequentially:

- Diff and annotate the modified code in the current and reference branches
- Checkout the reference branch, then run the workload the specified number of times
- Train an autoencoder on the saved performance counter data
- Reset the repository and re-checkout the first branch
- Run the workload again on the branch under test
- Flow the data through the autoencoder and flag anomalies in a report (:code:`.autoperf/report`).

.. warning::

    In your build process, be careful not to override any of the following environment variables:

    - :code:`LIBRARY_PATH`
    - :code:`LD_LIBRARY_PATH`
    - :code:`CPATH`
    - :code:`C_INCLUDE_PATH`
    - :code:`CPLUS_INCLUDE_PATH`
    - :code:`LIBS`
    - :code:`CFLAGS`
    - :code:`CPPFLAGS`
    - :code:`CXXFLAGS`

    AutoPerf uses these variables to dynamically link the PAPI library calls with the custom annotations. Appending is fine, but overwriting the above variables will likely break the build process.

3. **Tensorboard** (optional)

If you'd like to dig deeper into the internals of AutoPerf, we do expose a Tensorboard checkpoint that allows you to analyze different parts of the trained autoencoder. Simply type the following:

.. code-block:: console

    $ tensorboard --logdir .autoperf/logs/train/...

This will launch a Tensorboard server. If you open the page in your preferred web browser, you will be able to view the training progress of the autoencoder.

You can also analyze the latent space vectors of the nominal and test sets to identify their propensity towards clustering, through the use of the Embeddings Projector. This allows you to visualize the latent space in 2 or 3 dimensions using advanced dimensionality reduction algorithms like PCA, t-SNE, UMAP, and more.

4. :code:`clean` (optional)

If you'd like to return AutoPerf to the state it was in prior to running :code:`detect`, you can type:

.. code-block:: console

    $ autoperf clean

This will preserve your configuration files, but remove everything else under the :code:`.autoperf` directory.

☁️ CI/CD Pipeline
*******************

.. warning:: Not yet available. But, you could build one yourself fairly easily with a self-hosted runner!