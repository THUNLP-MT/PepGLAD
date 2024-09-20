#!/bin/bash
########## setup project directory ##########
CODE_DIR=`realpath $(dirname "$0")/..`
echo "Locate the project folder at ${CODE_DIR}"
cd ${CODE_DIR}

######### check number of args ##########
HELP="Usage example: GPU=0 bash $0 <name> <AE config> <LDM config> <latent dist config> <test config> [mode: e.g. 1111]"
if [ -z $1 ]; then
    echo "Experiment name missing. ${HELP}"
    exit 1;
else
    NAME=$1
fi
if [ -z $2 ]; then
    echo "Autoencoder config missing. ${HELP}"
    exit 1;
else
    AECONFIG=$2
fi
if [ -z $3 ]; then
    echo "LDM config missing. ${HELP}"
    exit 1;
else
    LDMCONFIG=$3
fi
if [ -z $4 ]; then
    echo "setup LDM dist config missing. ${HELP}"
    exit 1;
else
    LATENT_DIST_CONFIG=$4
fi
if [ -z $5 ]; then
    echo "LDM test config missing. ${HELP}"
    exit 1;
else
    TEST_CONFIG=$5
fi
if [ -z $6 ]; then
    MODE=1111
else
    MODE=$6
fi
echo "Mode: $MODE, [train AE] / [train LDM] / [Generate] / [Evalulation]"
TRAIN_AE_FLAG=${MODE:0:1}
TRAIN_LDM_FLAG=${MODE:1:1}
GENERATE_FLAG=${MODE:2:1}
EVAL_FLAG=${MODE:3:1}

AE_SAVE_DIR=./exps/$NAME/AE
LDM_SAVE_DIR=./exps/$NAME/LDM
OUTLOG=./exps/$NAME/output.log

if [[ ! -e ./exps/$NAME ]]; then
    mkdir -p ./exps/$NAME
elif [[ -e $AE_SAVE_DIR ]] && [ "$TRAIN_AE_FLAG" = "1" ]; then
    echo "Directory ${AE_SAVE_DIR} exisits! But training flag is 1!"
    exit 1;
elif [[ -e $LDM_SAVE_DIR ]] && [ "$TRAIN_LDM_FLAG" = "1" ]; then
    echo "Directory ${LDM_SAVE_DIR} exisits! But training flag is 1!"
    exit 1;
fi

########## train autoencoder ##########
echo "Training Autoencoder with config $AECONFIG:" > $OUTLOG
cat $AECONFIG >> $OUTLOG
if [ "$TRAIN_AE_FLAG" = "1" ]; then
    bash scripts/train.sh $AECONFIG --trainer.config.save_dir=$AE_SAVE_DIR
fi

########## train ldm ##########
echo "Training LDM with config $LDMCONFIG:" >> $OUTLOG
cat $LDMCONFIG >> $OUTLOG
AE_CKPT=`cat ${AE_SAVE_DIR}/version_0/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
echo "Using Autoencoder checkpoint: ${AE_CKPT}" >> $OUTLOG
if [ "$TRAIN_LDM_FLAG" = "1" ]; then
    bash scripts/train.sh $LDMCONFIG --trainer.config.save_dir=$LDM_SAVE_DIR --model.autoencoder_ckpt=$AE_CKPT
fi

########## get latent distance ##########
LDM_CKPT=`cat ${LDM_SAVE_DIR}/version_0/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
echo "Get distances in latent space" >> $OUTLOG
python setup_latent_guidance.py --config ${LATENT_DIST_CONFIG} --ckpt ${LDM_CKPT} --gpu ${GPU:0:1} >> $OUTLOG

########## generate ##########
echo "Generate results Using LDM checkpoint: ${LDM_CKPT}" >> $OUTLOG
if [ "$GENERATE_FLAG" = "1" ]; then
    python generate.py --config $TEST_CONFIG --ckpt $LDM_CKPT --gpu ${GPU:0:1}
fi

########## cal metrics ##########
if [ "$EVAL_FLAG" = "1" ]; then
    echo "Evaluation:" >> $OUTLOG
    python cal_metrics.py --results ${LDM_SAVE_DIR}/version_0/results/results.jsonl >> $OUTLOG
fi
