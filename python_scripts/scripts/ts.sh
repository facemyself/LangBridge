LOG_PATH="Logs"
TIME_STAMP=$(date "+%Y-%m-%d_%H-%M-%S")

LOG_NAME="Multilingual/MetaMath-qwen2.5"
#LOG_NAME="analyze/encoder/mt5-large"
LOG_PATH="$LOG_PATH/$LOG_NAME/$TIME_STAMP"

mkdir -p $LOG_PATH
ts --set_logdir $LOG_PATH

#ts -G 4 -L $LOG_NAME -O test_accelerate.log bash scripts/finetune.sh
#ts -G 8 -L $LOG_NAME -O llama_adapter_englishdata_4e-5and3e-5_bs128_0.05cosine.log bash scripts/finetune.sh

#ts -G 8 -L $LOG_NAME -O ALLE_X_CSQA.log bash scripts/training_x_csqa.sh
ts -G 4 -L $LOG_NAME -O MetaMath-qwen2.5.log bash scripts/train_lb/metamath.sh