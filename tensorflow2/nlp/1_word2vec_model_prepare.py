import codecs

import jieba.analyse

# 以写的方式打开原始的简体中文语料库
f = codecs.open('data/zhwiki_jian_zh_4.txt', 'r', encoding="utf8")
# 将分完词的语料写入到wiki_jian_zh_seg-189.5.txt文件中
target = codecs.open("data/wiki_jian_zh_4_seg.txt", 'w', encoding="utf8")

print('open files')
line_num = 1
line = f.readline()

# 循环遍历每一行，并对这一行进行分词操作
# 如果下一行没有内容的话，就会readline会返回-1，则while -1就会跳出循环
while line:
    print('---- processing ', line_num, ' article----------------')
    if line.startswith("</doc") or line.startswith("<doc"):
        line = f.readline()
        continue
    else:
        line_seg = " ".join(jieba.cut(line))
        target.writelines(line_seg)
        line_num = line_num + 1
        line = f.readline()

# 关闭两个文件流，并退出程序
f.close()
target.close()
exit()
