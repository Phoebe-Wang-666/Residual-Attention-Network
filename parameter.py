"""
In this parameter script,we mainly adjust the following
parameters:learning_rate(lrs),weight_decay(wds),batch_size(bss) and momentum(mom)

We use the grid search ideas: tring each combo of all hyperparameters one by one
and save the result into a csv file.

"""
import os
import json
import subprocess
# params
lrs=[0.1,0.05,0.01]      # learning rate
wds=[1e-4,5e-4]          # weight decay
bss=[64,128,256]         # batch size
mom=[0.8,0.9]        # momentum values 
train_py="train_cifar_tf.py"   # training script file
logdir="logs_tf"             # folder for json logs
csv_file="hyperparameter_result.csv"  # output of hyparameters
# write header directly 
with open(csv_file,"w") as f:
    f.write("lr,wd,bs,momentum,val,test,json\n")
for lr in lrs:
    for wd in wds:
        for bs in bss:
            for m in mom:
                # run training
                cmd="python "+train_py+\
                " --dataset cifar10"+\
                " --model attention56"+\
                " --att-type arl"+\
                " --epochs 10"+\
                " --lr "+str(lr)+\
                " --weight-decay "+str(wd)+\
                " --batch-size "+str(bs)+\
                " --momentum "+str(mom)

                print("run:",cmd)
                # run training
                os.system(cmd)
                # find json file (get the latest)
                logs = sorted(os.listdir(logdir))
                json_path = logs[-1]
                # read json
                with open(json_path) as f:
                  data = json.load(f)
                val=data.get("best_val_acc",0)
                test=data.get("test_acc",0)
                # append result
                with open(csv_file,"a") as f:
                  f.write(str(lr)+","+str(wd)+","+str(bs)+","+str(mom)+","+str(val)+","+str(test)+","+json_path+"\n")


