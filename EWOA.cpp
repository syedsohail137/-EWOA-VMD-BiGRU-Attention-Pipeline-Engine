#include "EWOA.h"
#include <vector>
#include <random>
#include <cmath>
#include <limits>

long Global_K = 2;
double Global_Alpha = 100.0;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double Evaluate_Fitness(long K, double Alpha, const std::vector<double>& Target, long N) {
    double MockSum = 0.0;
    for (long t = 0; t < N; ++t) {
        MockSum += (Target[t] * std::sin(static_cast<double>(t + 1) * static_cast<double>(K) / Alpha));
    }
    return std::abs(MockSum / static_cast<double>(N));
}

void Run_EWOA(const std::vector<double>& targetSeries, long N) {
    const long Max_Iter = 10;
    const long PopSize = 5;
    const long DimSize = 2; // 0: K, 1: Alpha
    
    std::vector<double> lb = { 2.0, 100.0 };
    std::vector<double> ub = { 6.0, 2000.0 };
    
    std::vector<std::vector<double>> Positions(PopSize, std::vector<double>(DimSize, 0.0));
    std::vector<double> Leader_Pos(DimSize, 0.0);
    double Leader_Score = std::numeric_limits<double>::infinity();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    
    for (long i = 0; i < PopSize; ++i) {
        Positions[i][0] = std::floor((ub[0] - lb[0] + 1.0) * dis(gen) + lb[0]);
        Positions[i][1] = lb[1] + (ub[1] - lb[1]) * dis(gen);
        
        double CurrentFitness = Evaluate_Fitness(static_cast<long>(Positions[i][0]), Positions[i][1], targetSeries, N);
        if (CurrentFitness < Leader_Score) {
            Leader_Score = CurrentFitness;
            Leader_Pos[0] = Positions[i][0];
            Leader_Pos[1] = Positions[i][1];
        }
    }
    
    double a = 0.0, a2 = 0.0, r1 = 0.0, r2 = 0.0, p = 0.0;
    double A_coef = 0.0, C_coef = 0.0, l = 0.0, distance = 0.0;
    
    for (long iter = 1; iter <= Max_Iter; ++iter) {
        a = 2.0 - static_cast<double>(iter) * (2.0 / static_cast<double>(Max_Iter));
        a2 = -1.0 + static_cast<double>(iter) * (-1.0 / static_cast<double>(Max_Iter));
        
        for (long i = 0; i < PopSize; ++i) {
            r1 = dis(gen);
            r2 = dis(gen);
            A_coef = 2.0 * a * r1 - a;
            C_coef = 2.0 * r2;
            p = dis(gen);
            l = (a2 - 1.0) * dis(gen) + 1.0;
            
            if (p < 0.5) {
                if (std::abs(A_coef) < 1.0) {
                    for (long j = 0; j < DimSize; ++j) {
                        distance = std::abs(C_coef * Leader_Pos[j] - Positions[i][j]);
                        Positions[i][j] = Leader_Pos[j] - A_coef * distance;
                    }
                } else {
                    long RandIndex = static_cast<long>(PopSize * dis(gen));
                    if (RandIndex >= PopSize) RandIndex = PopSize - 1;
                    for (long j = 0; j < DimSize; ++j) {
                        distance = std::abs(C_coef * Positions[RandIndex][j] - Positions[i][j]);
                        Positions[i][j] = Positions[RandIndex][j] - A_coef * distance;
                    }
                }
            } else {
                for (long j = 0; j < DimSize; ++j) {
                    distance = std::abs(Leader_Pos[j] - Positions[i][j]);
                    Positions[i][j] = distance * std::exp(0.5 * l) * std::cos(2.0 * M_PI * l) + Leader_Pos[j];
                }
            }
            
            if (Positions[i][0] < lb[0]) Positions[i][0] = lb[0];
            if (Positions[i][0] > ub[0]) Positions[i][0] = ub[0];
            Positions[i][0] = std::floor(Positions[i][0]);
            
            if (Positions[i][1] < lb[1]) Positions[i][1] = lb[1];
            if (Positions[i][1] > ub[1]) Positions[i][1] = ub[1];
            
            double CurrentFitness = Evaluate_Fitness(static_cast<long>(Positions[i][0]), Positions[i][1], targetSeries, N);
            if (CurrentFitness < Leader_Score) {
                Leader_Score = CurrentFitness;
                Leader_Pos[0] = Positions[i][0];
                Leader_Pos[1] = Positions[i][1];
            }
        }
    }
    Global_K = static_cast<long>(Leader_Pos[0]);
    Global_Alpha = Leader_Pos[1];
}
