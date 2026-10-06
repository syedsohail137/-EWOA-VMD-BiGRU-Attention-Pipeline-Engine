#ifndef EWOA_H
#define EWOA_H

#include <vector>
#include <string>

// Structure to hold raw row values from the CSV
struct RowData {
    std::string date;
    double open;
    double high;
    double low;
    double close;
    double volume;
};

// Global optimized parameters passed across modules
extern long Global_K;
extern double Global_Alpha;

// Module 1 & 2: Optimization and Signal Parsing
double Evaluate_Fitness(long K, double alpha, const std::vector<double>& targetSeries, long N);
void Run_EWOA(const std::vector<double>& targetSeries, long N);
void Execute_VMD(const std::vector<double>& InputSeries, long N, long K, double Alpha, std::vector<double>& OutputIMF);

// Module 3: Deep Neural Primitives with External Weights Routing
double Sigmoid(double x);
double Tanh(double x);
void Compute_BiGRU(const std::vector<std::vector<double>>& Features, long N, long InputDim, long HiddenDim, 
                   const std::vector<double>& W_fwd, const std::vector<double>& W_bwd,
                   std::vector<std::vector<double>>& CombinedH);

// Module 4: Master Orchestrator Pipeline
void Run_Main_EWOA_VMD_BiGRU_ATT_Pipeline(const std::string& csvFilePath);

#endif // EWOA_H
