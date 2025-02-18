import os

def fix_jpg_suffix(directory):
    """
    将目标目录内所有以.jpg.jpg结尾的文件名修正为.jpg后缀
    """
    for filename in os.listdir(directory):
        if filename.endswith('.jpg.jpg'):
            # 分割文件名和扩展名两次
            base_part = os.path.splitext(filename)[0]  # 第一次分割去掉最后一个.jpg
            new_base = os.path.splitext(base_part)[0]  # 第二次分割再去掉一个.jpg
            new_filename = f"{new_base}.jpg"
            
            # 构建完整文件路径
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            try:
                os.rename(old_path, new_path)
                print(f"成功重命名: {filename} -> {new_filename}")
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")
                os.remove(old_path)

if __name__ == "__main__":
    target_dir = "D:/mp16/downloads"
    
    # 检查目录是否存在
    if not os.path.exists(target_dir):
        print(f"错误：目录 {target_dir} 不存在")
    else:
        fix_jpg_suffix(target_dir)
        print("处理完成")