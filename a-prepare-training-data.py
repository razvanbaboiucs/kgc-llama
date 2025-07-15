import random
from string import Template

data_file = "data/FB13/train-minimal.tsv"
output_file = "data/FB13/train_minimal_instructions_llama_new.json"

prompt_template = Template("$head $relation $tail")
instruction_json_template = Template("""{
    "instruction": "Is the following knowledge graph triplet True or False?",
    "input": "$prompt",
    "output": "$output"
}
""")

ent2txt = {}

with open("data/FB13/entity2text_capital.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        ent2txt[tmp[0]] = tmp[1]
rel2txt = {}

with open("data/FB13/relation2text.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")
        rel2txt[tmp[0]] = tmp[1]

ent_list = []
for ent in ent2txt:
    ent_list.append(ent)

lines_to_write_llama_lora = []
with open(data_file, "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.strip().split("\t")

        prompt = prompt_template.safe_substitute(head=ent2txt[tmp[0]], relation=rel2txt[tmp[1]], tail=ent2txt[tmp[2]])
        tmp_str = instruction_json_template.safe_substitute(prompt=prompt, output=True)

        lines_to_write_llama_lora.append(tmp_str)

        rnd = random.random()

        if rnd <= 0.5:
            # corrupting head
            tmp_ent_list = set(ent_list)
            tmp_ent_list.remove(tmp[0])
            tmp_ent_list = list(tmp_ent_list)
            tmp_head = random.choice(tmp_ent_list)
            prompt = prompt_template.safe_substitute(head=ent2txt[tmp_head], relation=rel2txt[tmp[1]], tail=ent2txt[tmp[2]])
            tmp_str = instruction_json_template.safe_substitute(prompt=prompt, output=False)
            lines_to_write_llama_lora.append(tmp_str)

        else:
            # corrupting tail
            tmp_ent_list = set(ent_list)
            tmp_ent_list.remove(tmp[2])
            tmp_ent_list = list(tmp_ent_list)
            tmp_tail = random.choice(tmp_ent_list)
            prompt = prompt_template.safe_substitute(head=ent2txt[tmp[0]], relation=rel2txt[tmp[1]], tail=ent2txt[tmp_tail])
            tmp_str = instruction_json_template.safe_substitute(prompt=prompt, output=False)
            lines_to_write_llama_lora.append(tmp_str)

with open(output_file, "w") as f:
    tmp_str = "[\n" + ",\n".join(lines_to_write_llama_lora) + "]"
    f.write(tmp_str)
