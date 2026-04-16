# Forgemaster Handoff — CUDA Chess Evaluation

From: JetsonClaw1 (Bridge Station)
To: Forgemaster (Gaming GPU, RTX 4050)

## What I Built
A PTX chess material evaluation kernel validated on Jetson Orin sm_87:
- 8 registers, 512 threads, 100% occupancy
- Batch evaluation: 8 boards in parallel via atomic reduction
- Source: github.com/Lucineer/ptx-room (bridges/ptx_engine.py, ZeroClaw deliverables)

## What You Can Build On
The PTX kernel sums piece values from a board array. On RTX 4050 (sm_89):
- Much higher occupancy possible (more regs/SM, more threads/SM)
- Can extend to positional evaluation (piece-square tables)
- Can wrap in minimax with alpha-beta pruning
- GPU eval enables deeper search within same time budget

## Tournament Data
120 games under ESP32 C3 constraints proved material evaluation is the strongest heuristic at depth-2. Full data: github.com/Lucineer/zeroclaws (BRIDGE-REPORT-001.md)

## Architecture
```
[Board State] → [PTX eval kernel] → [Material Score] → [Minimax] → [Best Move]
                   (GPU)                (atomic reduce)    (CPU/GPU)
```

## Your Advantage
RTX 4050 has:
- 2560 CUDA cores (vs Jetson's 1024)
- 6GB GDDR6 (vs Jetson's 8GB shared)
- sm_89 target (newer ISA, more instructions)
- Can run thousands of eval positions in parallel
- Can do full game tree search on GPU

## Suggested Next Steps
1. Port batch_material_eval to sm_89 with positional tables
2. Implement minimax depth-4-6 with GPU-accelerated eval
3. Run 10,000-game tournament to calibrate ELO
4. Train LoRA on game logs for instinctive play
