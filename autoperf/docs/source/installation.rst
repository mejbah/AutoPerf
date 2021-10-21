Installation
============

Here's how you can get started using AutoPerf on your software repository!

.. tip::
    For AutoPerf to work seamlessly, you must be using Linux (preferably Ubuntu 20.04) with the ability to `sudo` with a non-virtualized Intel CPU (though you can use :ref:`Docker<🐳 Use Docker (Optional)>`!)

👥 Clone Repository
*******************

Clone the `GitHub repository <https://github.com/IntelLabs/machine_programming_research/tree/master>`_
to your local machine.

.. code-block:: console

    $ git clone git@github.com:IntelLabs/machine_programming_research.git

⏱️ Install PAPI
***************

Use your preferred package manager to install the performance application programming interface (PAPI) libraries.
At a minimum, you need the base library (`libpapi`) and the development headers `libpapi-dev`. The preferred
version is 5.7.

Some repositories also include `papi-tools`, which exposes helpful utilities like `papi-avail`. This particular
executable lists all available performance counter events for your particular hardware.

All of this can be done via Aptitude:

.. code-block:: console

    $ sudo apt install libpapi-dev libpapi5.7 papi-tools

🐍 Install Conda
****************

Install `Anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_ or
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

Next, clone the AutoPerf Conda environment:

.. code-block:: console

    $ conda env create -f environment_ubuntu.yml

If you're not using Ubuntu, you can try using the base `environment.yml` file,
but it will take longer to solve the environment, sometimes up to 10 minutes.

Once the environment is created, activate it by typing:

.. code-block:: console

    $ conda activate ap

📦 Install AutoPerf
*******************

AutoPerf is not yet available on the Python Package Index, so it must be installed
manually at this time. You can do so by entering the base of the repository and typing:

.. code-block:: console

    $ python setup.py install

This will build the required C dynamic library and place AutoPerf in your `PYTHONPATH`.

Once installed, the AutoPerf CLI can be accessed by typing:

.. code-block:: console

    $ autoperf --help

⚙️ Configure System
*******************

Before AutoPerf can analyze your code, you need to allow unprivileged users access
to performance counter events. This can be done via the following line.

.. code-block:: console

    $ sudo sh -c 'echo -1 >/proc/sys/kernel/perf_event_paranoid'

🐳 Use Docker (Optional)
************************

At the base level of the repository, type:

.. code-block:: console

    $ docker build

Interacting with the container itself is out of scope for this installation guide, but there
is one aspect that should be discussed here.

As mentioned previously, the kernel's `perf_event_paranoid` flag must be set to -1 for AutoPerf to
interact with the hardware performance counters. By default, Docker has disabled this behavior due
to a potential security risk (tracing/profiling syscalls can leak information about the host).

There are two options to get around this:

1. Follow the advice given `here <https://stackoverflow.com/a/44748260>`_ regarding a custom seccomp file that allows only the `perf_event_open` syscall.
2. Run the container with the `\-\-privileged` flag, which opens up further security vulnerabilities.