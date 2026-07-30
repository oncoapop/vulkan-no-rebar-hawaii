# CPU Baseline Report

**Environment:**
*   OS: Ubuntu 24.04
*   CPU: Intel Xeon E5-2699 v3, 12 cores, SMT disabled
*   RAM: 32 GB ECC
*   Model: GLM-5.2 Colibrì int4 gs64 with int8 MTP head (~372 GB on NVMe)

**Command to execute on Damian's Server:**
`./colibri --model /path/to/model -t 12 --temp 0 --draft 0 -p "Explain quantum computing."`

**Results:**
*   **Startup Time:** NOT_RUN (Pending hardware validation)
*   **Time to First Token (TTFT):** NOT_RUN
*   **Prefill Performance:** NOT_RUN
*   **Decode Performance:** NOT_RUN
*   **Total RAM Usage:** NOT_RUN
*   **Disk Throughput:** NOT_RUN
*   **Generated Text:** NOT_RUN

*Note: Baseline measurements must be collected on Damian's server as the Jules environment lacks the R9 390 and the 372 GB model.*
