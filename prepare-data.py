import random
from string import Template

prompt_template = Template("$head $relation $tail")
instruction_json_template = Template("""{
    "instruction": "Is the following knowledge graph triplet True or False?",
    "input": "$prompt",
    "output": "$output"
}
""")


def load_mapping(path):
    mapping = {}
    with open(path, "r") as f:
        for line in f.readlines():
            tmp = line.strip().split("\t")
            mapping[tmp[0]] = tmp[1]
    return mapping


def prepare_data(data_file, output_file, ent2txt, rel2txt, corrupt):
    ent_list = list(ent2txt)

    lines_to_write_llama_lora = []
    with open(data_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split("\t")

            prompt = prompt_template.safe_substitute(
                head=ent2txt[tmp[0]], relation=rel2txt[tmp[1]], tail=ent2txt[tmp[2]]
            )

            if corrupt:
                # positive triplet
                tmp_str = instruction_json_template.safe_substitute(prompt=prompt, output=True)
                lines_to_write_llama_lora.append(tmp_str)

                rnd = random.random()

                if rnd <= 0.5:
                    # corrupting head
                    tmp_ent_list = set(ent_list)
                    tmp_ent_list.remove(tmp[0])
                    tmp_ent_list = list(tmp_ent_list)
                    tmp_head = random.choice(tmp_ent_list)
                    prompt = prompt_template.safe_substitute(
                        head=ent2txt[tmp_head], relation=rel2txt[tmp[1]], tail=ent2txt[tmp[2]]
                    )
                    tmp_str = instruction_json_template.safe_substitute(prompt=prompt, output=False)
                    lines_to_write_llama_lora.append(tmp_str)

                else:
                    # corrupting tail
                    tmp_ent_list = set(ent_list)
                    tmp_ent_list.remove(tmp[2])
                    tmp_ent_list = list(tmp_ent_list)
                    tmp_tail = random.choice(tmp_ent_list)
                    prompt = prompt_template.safe_substitute(
                        head=ent2txt[tmp[0]], relation=rel2txt[tmp[1]], tail=ent2txt[tmp_tail]
                    )
                    tmp_str = instruction_json_template.safe_substitute(prompt=prompt, output=False)
                    lines_to_write_llama_lora.append(tmp_str)
            else:
                # label is read directly from the data file
                output = tmp[3] == "1"
                tmp_str = instruction_json_template.safe_substitute(prompt=prompt, output=output)
                lines_to_write_llama_lora.append(tmp_str)

    with open(output_file, "w") as f:
        tmp_str = "[\n" + ",\n".join(lines_to_write_llama_lora) + "]"
        f.write(tmp_str)


def main():
    ent2txt = load_mapping("data/FB13/entity2text_capital.txt")
    rel2txt = load_mapping("data/FB13/relation2text.txt")

    datasets = [
        # data_file, output_file, corrupt
        ("data/FB13/train-minimal.tsv", "data/FB13/train_minimal_instructions_llama_new.json", True),
        ("data/FB13/dev-minimal.tsv", "data/FB13/dev_minimal_instructions_llama_new.json", True),
        ("data/FB13/test.tsv", "data/FB13/test_instructions_llama_new.json", False),
    ]

    for data_file, output_file, corrupt in datasets:
        prepare_data(data_file, output_file, ent2txt, rel2txt, corrupt)


if __name__ == "__main__":
    main()
