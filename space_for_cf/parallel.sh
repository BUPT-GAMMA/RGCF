CONFIG_DIR=$1
REPEAT=$2
MAX_JOBS=${3:-2}
SLEEP=${4:-1}
GPU_STRATEGY=${5}

#[ -e ./fd1 ] || mkfifo ./fd1
#exec 3<> ./fd1
#rm -rf ./fd1
#
#for i in `seq 1 $MAX_JOBS`;
#do
#    echo >&3
#done


for CONFIG in "$CONFIG_DIR"/*.yaml;
do
#    echo $CONFIG
#    python main.py --cfg $CONFIG --repeat $REPEAT --mark_done --gpu_strategy $GPU_STRATEGY
#    read -u3
#    {
      python main.py --cfg $CONFIG --repeat $REPEAT --mark_done --gpu_strategy $GPU_STRATEGY

        sleep $SLEEP
#        echo >&3
#    }
done
)

#exec 3<&-
#exec 3>&-