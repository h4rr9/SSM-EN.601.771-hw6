# hparam search over lr and nepochs

for nepoch in 5 7 9
do
    for lr in 1e-4 5e-4 1e-5
    do
        sbatch launchpad_params.sh $nepoch $lr
    done
done
