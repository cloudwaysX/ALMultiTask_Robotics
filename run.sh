
# Pendulum
for seed in 425 500 987 26794 56 89 1111 89731 21312
do
    python test_pendulum_nonlinear.py --config config1 --seed $seed
    python test_pendulum_nonlinear.py --config config2 --seed $seed
done

# Synthetic
for seed in 425 500 987 26794 56 89 1111 89731 21312
do
    python test_linear.py --config linear_config1 --data_seed $seed
    python test_linear.py --config linear_config2 --data_seed $seed
    python test_linear.py --config linear_config3 --data_seed $seed
    python test_linear.py --config linear_config4 --data_seed $seed
    python test_linear.py --config linear_config5 --data_seed $seed
    python test_linear.py --config linear_config6 --data_seed $seed
done

for seed in 425 500 987 26794 56 89 1111 89731 21312
do
    python test_nonlinearKernel.py --config linear_config1 --data_seed $seed
    python test_nonlinearKernel.py --config linear_config2 --data_seed $seed
    python test_nonlinearKernel.py --config linear_config3 --data_seed $seed
    python test_nonlinearKernel.py --config linear_config4 --data_seed $seed
    python test_nonlinearKernel.py --config linear_config5 --data_seed $seed
    python test_nonlinearKernel.py --config linear_config6 --data_seed $seed
done


for seed in 425 500 987 26794 56 89 1111 89731 21312
do
    python test_shallow.py --config shallow_config1 --data_seed $seed
    python test_shallow.py --config shallow_config2 --data_seed $seed
    python test_shallow.py --config shallow_config3 --data_seed $seed
    python test_shallow.py --config shallow_config4 --data_seed $seed
    python test_shallow.py --config shallow_config5 --data_seed $seed
    python test_shallow.py --config shallow_config6 --data_seed $seed
done

# Drone passive and active
python test_drone_passive.py 
python test_drone_active.py 