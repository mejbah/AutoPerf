Virtualization with VMWare
==================================
VMWare Workstation 16 with Ubuntu 20.04 LTS x86_64 guest

- On the Virtual Hardware tab, expand CPU and select the 'Enable virtualized CPU performance counters' check-box.
- Flag 'vpmc.freezeMode=guest' such that all events increment only when guest instructions are directly executing on a physical CPU.
- Installation on Linux 5.4.0-77-generic Ubuntu 20.04 LTS x86_64
    - Note: Review literature for complex cases information [#papiV]_ [#papiVMC]_ [#papiVMCC]_ [#papiVMP]_ [#papiVMSDK]_ [#papiVMG]_

.. code-block:: console

    jdtarang@skynetAI:~$ uname -a
        Linux skynetAI 5.4.0-77-generic #86-Ubuntu SMP Thu Jun 17 02:35:03 UTC 2021 x86_64 x86_64 x86_64 GNU/Linux
    jdtarang@skynetAI:~$ sudo apt-get install linux-tools-common fakeroot libncurses5-dev

References
*************
.. [#papiV] Johnson et al., "PAPI-V: Performance Monitoring for Virtual Machines," 2012 41st International Conference on Parallel Processing Workshops, 2012, pp. 194-199, doi: 10.1109/ICPPW.2012.29.
.. [#papiVMC] PAPI for VMWare Components. https://bitbucket.org/icl/papi/src/master/src/components/vmware/README.md
.. [#papiVMCC] PAPI VMWare configuration. https://bitbucket.org/icl/papi/src/master/src/components/vmware/PAPI-VMwareComponentDocument.pdf
.. [#papiVMP] VMWare, Virtualization of Processor Performance on Guest. https://kb.vmware.com/s/article/2030221
.. [#papiVMSDK] VMware: http://www.vmware.com/support/developer/guest-sdk.
.. [#papiVMG] VMWare Workstation Guest. https://escalatingtechie.blogspot.com/2012/10/papi-install-guide.html