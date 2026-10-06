#include "EWOA.h"
#include <iostream>

int main() {
    std::string csvPath = "Daily_EQ_Data.csv";
    std::cout << "[SYSTEM] Initializing Optimized EWOA-VMD-BiGRU Execution Workspace...\n";
    Run_Main_EWOA_VMD_BiGRU_ATT_Pipeline(csvPath);
    return 0;
}
