#include "EWOA.h"
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void Execute_VMD(const std::vector<double>& InputSeries, long N, long K, double Alpha, std::vector<double>& OutputIMF) {
    OutputIMF.resize(N, 0.0);
    double Omega = 0.05;
    
    for (long t = 0; t < N; ++t) {
        double CombinedModeValue = 0.0;
        for (long k_idx = 1; k_idx <= K; ++k_idx) {
            double angle = 2.0 * M_PI * Omega * static_cast<double>(t + 1) * static_cast<double>(k_idx) / Alpha;
            CombinedModeValue += (InputSeries[t] * std::cos(angle));
        }
        OutputIMF[t] = CombinedModeValue / static_cast<double>(K);
    }
}
