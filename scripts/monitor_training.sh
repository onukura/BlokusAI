#!/bin/bash
# Monitor ongoing training progress

echo "=== Training Monitor ==="
echo ""

# Check for running training processes
echo "Running training processes:"
ps aux | grep -E "(train|python)" | grep -v grep | head -5
echo ""

# Check latest training log
if [ -f "medium_training.log" ]; then
    echo "=== Latest training output ==="
    tail -30 medium_training.log
elif [ -f "training_log.txt" ]; then
    echo "=== Training log ==="
    tail -30 training_log.txt
else
    echo "No training log found yet"
fi

echo ""
echo "=== Model files ==="
ls -lh *.pth 2>/dev/null || echo "No model files found"

echo ""
echo "=== Visualization outputs ==="
ls -lh *.png 2>/dev/null | head -5
ls -lh game_analysis/*.png 2>/dev/null | wc -l | xargs echo "Game analysis images:"
