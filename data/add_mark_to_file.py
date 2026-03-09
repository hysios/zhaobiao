def add_mark_to_file(input_file, output_file, mark="&&&&"):
    """
    给文件每行末尾添加指定标记
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        mark: 要添加的标记，默认为"&&&&"
    """
    try:
        # 读取输入文件内容
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 给每行添加标记
        marked_lines = [line.rstrip('\n') + mark + '\n' for line in lines]
        
        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(marked_lines)
        
        print(f"处理完成！已将标记'{mark}'添加到每行末尾，结果保存至{output_file}")
    
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
    except Exception as e:
        print(f"处理过程中发生错误：{e}")

if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = "中华人民共和国招标投标法.jsonl"
    output_file = "中华人民共和国招标投标法_标记版.jsonl"
    
    # 调用函数处理文件
    add_mark_to_file(input_file, output_file)

    # 输入和输出文件路径
    input_file = "中华人民共和国招标投标法实施条例.jsonl"
    output_file = "中华人民共和国招标投标法实施条例_标记版.jsonl"
    
    # 调用函数处理文件
    add_mark_to_file(input_file, output_file)