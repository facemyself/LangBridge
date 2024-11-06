from transformers import AutoTokenizer
from langbridge import LangBridgeModel

# our pretrained langbridge models all leverage this encoder tokenizer
enc_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct') 
lm_tokenizer = AutoTokenizer.from_pretrained('meta-math/MetaMath-7B-V1.0')
model = LangBridgeModel.from_pretrained('/workspace/LangBridge/python_scripts/checkpoints/metamath-qwen2.5-stage3-6-6-25-21/epoch=1').half().to('cuda')

enc_tokenizer.bos_token = enc_tokenizer.pad_token
enc_tokenizer.bos_token_id = enc_tokenizer.pad_token_id
metamath_template = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
    )
question1 = "O humano tinha necessidades luxuriosas a satisfazer, para onde foi ele em resultado disso?"
prompt1 = metamath_template.format(instruction=question1)
question2 = "珍妮特的鸭子每天下 16 颗蛋。她每天早上早餐时吃 3 颗，每天用 4 颗为自己的朋友做松饼。剩下的鸭蛋她每天拿去农贸市场卖，每颗新鲜鸭蛋卖 2 美元。她每天在农贸市场赚多少钱？"
prompt2 = metamath_template.format(instruction=question2)
output = model.generate_from_prefix(enc_tokenizer, prompts=[prompt1, prompt2])
print(output)