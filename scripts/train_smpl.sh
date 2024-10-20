EXPID=$1
CONFIG=$2
PORT=${3:-23456}

HOST=$(hostname -i)

echo ${CONFIG}

python ./scripts/train_smpl.py
    --nThreads 16
    --launcher pytorch --rank 0
    --dist-url tcp://${HOST}:${PORT}
    --exp-id ${EXPID}
    --cfg ${CONFIG} --seed 123123
