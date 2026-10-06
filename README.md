# EWOA-VMD-BiGRU+Attention Pipeline Engine



## Project Structure
- `src/EWOA.h`: Main structural header file mapping functions, global variables, and data frames.
- `src/EWOA.cpp`: Module 1 (Enhanced Whale Optimization Algorithm) parameter discoverer.
- `src/VMD.cpp`: Module 2 (Variational Mode Decomposition emulation layer).
- `src/BiGRU.cpp`: Module 3 (Bidirectional Gated Recurrent Unit network and activation primitives).
- `src/Pipeline.cpp`: Module 4 (Pipeline orchestrator, VWAP generator, and file system output manager).
- `src/main.cpp`: Execution entry point.

## How to Build & Run
Ensure you have a standard C++ compiler (supporting C++11 or higher) installed. Place a historical file named `Daily_EQ_Data.csv` in the execution directory (the CSV must contain at least 128 rows of history with data columns ordered as: `Date,Open,High,Low,Close,Volume`).

### Compilation Command (GCC/Clang):
```bash
g++ -O3 src/*.cpp -o quant_engine
```

### Run the Engine:
```bash
./quant_engine
```

Upon execution, the code analyzes the window history, discovers optimal configurations, prints `T+1` and `T+2` out-of-sample forward projections to the terminal, and flushes full structured matrix metrics into a newly created spreadsheet-ready file named `Pipeline_Output.csv`.

To build and compile these raw files directly in your command line terminal workspace using any standard GCC/G++ distribution tool, drop all these files into a single folder and execute the following line:



g++ -O3 main.cpp Pipeline.cpp EWOA.cpp VMD.cpp BiGRU.cpp -o HybridForecastEngine

(Note: The -O3 modifier tells the compiler to aggressively optimize matrix execution speed.)

Date,Open,High,Low,Close,Volume
2026-01-02,150.25,152.40,149.80,151.10,1250000
2026-01-03,151.10,153.85,150.50,153.20,1420000
... (Must contain at least 128 rows)
