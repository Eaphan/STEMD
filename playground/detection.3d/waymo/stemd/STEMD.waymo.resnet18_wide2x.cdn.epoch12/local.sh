function check_and_do(){
    count="999"
    while [ $count != "0" ];do
        count=`ps -ef |grep $1 |grep -v "grep"|grep -E "yifan|yzhang3362" |wc -l`

        # ad hoc
        # count=`ps -ef |grep -E "pointpillar_B07|pointpillar_B08" |grep -v "grep"|grep "yifanzh" |wc -l`
        sleep 5
    done
    while ! efg_run --num-gpus 4  dataloader.batch_size 2 &>> log.txt;do sleep 2; done

}

check_and_do "efg_run"

