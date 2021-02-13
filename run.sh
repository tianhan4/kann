export OMP_NUM_THREADS=10
#rm /data/glusterfs/home/htianab/kann-data/batch-xxx # to rebuild the engine
./examples/layer_timer ./examples/kann-data/mnist-train-x.knd.gz ./examples/kann-data/mnist-train-y.knd.gz /data/glusterfs/home/htianab/kann-data > output 2> error
