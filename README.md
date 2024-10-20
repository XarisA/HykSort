# HykSort

HykSort Algorithm Implementation with ParallelSelect algorithm for finding splitters and AlltoAll_kway for all to all communication between processors, having as a result reduced complexity to O(N log p / log k).

A big part of the logic in this code was found in this project [2] and was rewritten in c++.
I added some helper functions and distinguished the HykSort algorithm from [2] so that it can run independenly.

**References**

1. Hari Sundar, Dhairya Malhotra, George Biros, [HykSort: a new variant of hypercube quicksort on distributed memory architectures](http://dx.doi.org/10.1145/2464996.2465442), Proceedings of the 27th international ACM conference on international conference on supercomputing (**ICS13**), 2013. 

2. Utah Sorting Librrary : https://github.com/hsundar/usort

3. Distributed Bitonic Sort using MPI : https://github.com/steremma/Bitonic
