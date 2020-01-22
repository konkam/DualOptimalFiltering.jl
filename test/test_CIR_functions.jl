
@test_nowarn DualOptimalFiltering.generate_CIR_trajectory(range(0, stop = 2, length = 20), 3, 3., 0.5, 1);
@test_nowarn DualOptimalFiltering.rCIR(2, 0.5, 0.5, 0.5, 0.5, 0.5)
