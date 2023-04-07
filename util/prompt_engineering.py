def generate_prompt(data_point):
    if data_point["instruction"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                    ### Instruction:
                    {data_point["instruction"]}

                    ### Input:
                    {data_point["input"]}

                    ### Response:
                    {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                    ### Instruction:
                    {data_point["instruction"]}

                    ### Response:
                    {data_point["output"]}"""
