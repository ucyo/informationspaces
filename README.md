Code and results for the publication [@IEEE Big Data 2018](http://cci.drexel.edu/bigdata/bigdata2018/index.html):
Concept and Analysis of Information Spaces to Improve Prediction-Based Compression.

# informationspaces
This repository extends the [compression framework](https://github.com/ucyo/cframework) introducted at ACM SIGSPATIAL 2018.


## Requirements and installation
This code has been tested on following machine:

```
Python: 3.6.1
OS: Debian 4.11.6-1 (2017-06-19) testing (buster)
CPU: Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz
MEM: 16 GiB 2400MHz DDR4
```

To recreate the software environment install using `pip`:

```bash
pip install -r requirements.txt
```

## Project structure & extensions
The `pasc` folder as a similiar structure as `cframe` in [compression framework](https://github.com/ucyo/cframework) with several extensions. For details please refer to the paper mentioned above:

- Additional sequencer are added to the repository
    - `chequerboard`
    - `blossom`
    - `blocks`
- Additional predictors are added
    - `pascal`
    - `ratana`
    - ...
- Consolidation methods are added using `toolbox.manager`
    - `MinManager`
    - `MaxManager`
    - `ReproduceManager`
    - `LastBestManager`
- Benchmarking script was added `benchmark.py`
