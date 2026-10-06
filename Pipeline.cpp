#include "EWOA.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

void Run_Main_EWOA_VMD_BiGRU_ATT_Pipeline(const std::string& csvFilePath) {
    const long TargetLength = 128;
    std::vector<RowData> allRecords;
    
    std::ifstream file(csvFilePath);
    if (!file.is_open()) {
        std::cerr << "Error: File '" << csvFilePath << "' could not be opened.\n";
        return;
    }
    
    std::string line;
    std::getline(file, line); // Skip CSV Header Row
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        RowData row;
        if (std::getline(ss, row.date, ',')) {
            if (std::getline(ss, cell, ',')) row.open = std::stod(cell);
            if (std::getline(ss, cell, ',')) row.high = std::stod(cell);
            if (std::getline(ss, cell, ',')) row.low = std::stod(cell);
            if (std::getline(ss, cell, ',')) row.close = std::stod(cell);
            if (std::getline(ss, cell, ',')) row.volume = std::stod(cell);
            allRecords.push_back(row);
        }
    }
    file.close();
    
    long TotalRows = static_cast<long>(allRecords.size());
    if (TotalRows < TargetLength) {
        std::cerr << "Data mismatch: requires 128 rows, found " << TotalRows << "\n";
        return;
    }
    
    long StartIdx = TotalRows - TargetLength;
    std::string LastDate = allRecords[TotalRows - 1].date;
    
    std::vector<double> ClosePrices(TargetLength, 0.0);
    std::vector<std::vector<double>> FeatureMatrix(TargetLength, std::vector<double>(3, 0.0));
    
    double CumVolume = 0.0;
    double CumTypicalPV = 0.0;
    
    for (long t = 0; t < TargetLength; ++t) {
        const auto& targetRow = allRecords[StartIdx + t];
        ClosePrices[t] = targetRow.close;
        double TypicalPrice = (targetRow.high + targetRow.low + targetRow.close) / 3.0;
        CumVolume += targetRow.volume;
        CumTypicalPV += (TypicalPrice * targetRow.volume);
        
        if (CumVolume > 0.0) {
            FeatureMatrix[t][0] = CumTypicalPV / CumVolume; // VWAP
        } else {
            FeatureMatrix[t][0] = ClosePrices[t];
        }
        FeatureMatrix[t][1] = targetRow.volume;
        FeatureMatrix[t][2] = ClosePrices[t];
    }
    
    // Core Processing Execution Chain
    Run_EWOA(ClosePrices, TargetLength);
    
    std::vector<double> DominantIMF;
    Execute_VMD(ClosePrices, TargetLength, Global_K, Global_Alpha, DominantIMF);
    
    std::vector<double> W_fwd = { 0.35, 0.08, 0.52 };
    std::vector<double> W_bwd = { -0.21, 0.04, -0.41 };
    std::vector<std::vector<double>> HiddenStates;
    Compute_BiGRU(FeatureMatrix, TargetLength, 3, 8, W_fwd, W_bwd, HiddenStates);
    
    // Mathematically Stable Softmax Shift Engine
    std::vector<double> NetScores(TargetLength, 0.0);
    double MaxScore = -std::numeric_limits<double>::infinity();
    for (long t = 0; t < TargetLength; ++t) {
        double Score = 0.0;
        for (long h = 0; h < 16; ++h) {
            Score += (HiddenStates[t][h] * 0.06);
        }
        Score += (DominantIMF[t] * 0.02);
        NetScores[t] = Score;
        if (Score > MaxScore) MaxScore = Score;
    }
    
    std::vector<double> AttWeights(TargetLength, 0.0);
    double AttSum = 0.0;
    for (long t = 0; t < TargetLength; ++t) {
        AttWeights[t] = std::exp(NetScores[t] - MaxScore); // Stable shift subtraction
        AttSum += AttWeights[t];
    }
    
    std::vector<std::vector<double>> DeliveryMatrix(TargetLength, std::vector<double>(5, 0.0));
    for (long t = 0; t < TargetLength; ++t) {
        double NormalizedAttention = (AttSum > 0.0) ? (AttWeights[t] / AttSum) : (1.0 / TargetLength);
        DeliveryMatrix[t][0] = FeatureMatrix[t][0];
        DeliveryMatrix[t][1] = static_cast<double>(Global_K);
        DeliveryMatrix[t][2] = Global_Alpha;
        DeliveryMatrix[t][3] = NormalizedAttention;
        DeliveryMatrix[t][4] = FeatureMatrix[t][2] * (1.0 + (NormalizedAttention * 0.015));
    }
    
    double FinalAttention = DeliveryMatrix[TargetLength - 1][3];
    double FinalPrediction = DeliveryMatrix[TargetLength - 1][4];
    
    double T1_Forecast = FinalPrediction * (1.0 + (std::sin(static_cast<double>(TargetLength + 1) * static_cast<double>(Global_K) / Global_Alpha) * FinalAttention * 0.5));
    double T2_Forecast = T1_Forecast * (1.0 + (std::sin(static_cast<double>(TargetLength + 2) * static_cast<double>(Global_K) / Global_Alpha) * FinalAttention * 0.5));
    
    // Flush Output Matrix Array out to Flat CSV File
    std::string outFilePath = "Pipeline_Output.csv";
    std::ofstream outFile(outFilePath);
    if (!outFile.is_open()) return;
    
    outFile << "Date,Open,High,Low,Close,Volume,VWAP,Discovered_K,Discovered_Alpha,Attention_Weight,Prediction_Label,Value\n";
    for (long t = 0; t < TargetLength; ++t) {
        const auto& rawRow = allRecords[StartIdx + t];
        outFile << rawRow.date << "," << rawRow.open << "," << rawRow.high << "," << rawRow.low << "," << rawRow.close << "," << rawRow.volume << ","
                << DeliveryMatrix[t][0] << "," << DeliveryMatrix[t][1] << "," << DeliveryMatrix[t][2] << "," << DeliveryMatrix[t][3] << ",Historical_Prediction," << DeliveryMatrix[t][4] << "\n";
    }
    outFile << LastDate << "+1_Day,0,0,0,0,0," << DeliveryMatrix[TargetLength - 1][0] << "," << Global_K << "," << Global_Alpha << ",0,T+1 Forecast," << T1_Forecast << "\n";
    outFile << LastDate << "+2_Days,0,0,0,0,0," << DeliveryMatrix[TargetLength - 1][0] << "," << Global_K << "," << Global_Alpha << ",0,T+2 Forecast," << T2_Forecast << "\n";
    outFile.close();
    
    std::cout << "\n==================================================\n";
    std::cout << "Pipeline completed successfully! Saved to: " << outFilePath << "\n";
    std::cout << "T+1 Forecast: " << T1_Forecast << "\n";
    std::cout << "T+2 Forecast: " << T2_Forecast << "\n";
    std::cout << "==================================================\n";
}
