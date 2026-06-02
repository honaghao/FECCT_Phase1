#!/bin/bash
echo "======================================================================"
echo "Phase 1: Minimal 3x3 Transfer Matrix Training"
echo "======================================================================"
echo ""
echo "This will train 9 models (3 codes x 3 seeds)"
echo "Estimated time: 4-5 days on RTX 4070 16GB"
echo ""
echo "Press Enter to continue, or Ctrl+C to cancel..."
read

python train_phase1.py

echo ""
echo "======================================================================"
echo "Training complete! Run evaluate_phase1.py to build transfer matrix."
echo "======================================================================"
