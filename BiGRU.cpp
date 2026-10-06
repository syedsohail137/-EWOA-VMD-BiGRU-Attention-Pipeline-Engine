#include "EWOA.h"
#include <cmath>
#include <vector>

double Sigmoid(double x) {
    if (x > 20.0) return 1.0;
    if (x < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

double Tanh(double x) {
    if (x > 20.0) return 1.0;
    if (x < -20.0) return -1.0;
    double e2x = std::exp(2.0 * x);
    return (e2x - 1.0) / (e2x + 1.0);
}

void Compute_BiGRU(const std::vector<std::vector<double>>& Features, long N, long InputDim, long HiddenDim, 
                   const std::vector<double>& W_fwd, const std::vector<double>& W_bwd,
                   std::vector<std::vector<double>>& CombinedH) {
    
    std::vector<std::vector<double>> ForwardH(N, std::vector<double>(HiddenDim, 0.0));
    std::vector<std::vector<double>> BackwardH(N, std::vector<double>(HiddenDim, 0.0));
    CombinedH.assign(N, std::vector<double>(HiddenDim * 2, 0.0));
    
    // Forward Pass
    for (long t = 0; t < N; ++t) {
        for (long h = 0; h < HiddenDim; ++h) {
            double FwdNetInput = 0.0;
            for (long f = 0; f < InputDim; ++f) {
                FwdNetInput += (Features[t][f] * W_fwd[f]);
            }
            if (t == 0) {
                ForwardH[t][h] = Tanh(FwdNetInput);
            } else {
                ForwardH[t][h] = Tanh(FwdNetInput + (ForwardH[t - 1][h] * 0.15));
            }
        }
    }
    
    // Backward Pass
    for (long t = N - 1; t >= 0; --t) {
        for (long h = 0; h < HiddenDim; ++h) {
            double BwdNetInput = 0.0;
            for (long f = 0; f < InputDim; ++f) {
                BwdNetInput += (Features[t][f] * W_bwd[f]);
            }
            if (t == N - 1) {
                BackwardH[t][h] = Tanh(BwdNetInput);
            } else {
                BackwardH[t][h] = Tanh(BwdNetInput + (BackwardH[t + 1][h] * 0.15));
            }
        }
    }
    
    // Layout Merge
    for (long t = 0; t < N; ++t) {
        for (long h = 0; h < HiddenDim; ++h) {
            CombinedH[t][h] = ForwardH[t][h];
            CombinedH[t][h + HiddenDim] = BackwardH[t][h];
        }
    }
}
