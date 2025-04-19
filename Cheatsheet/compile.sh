#!/bin/bash

> source/Cheatsheet-All.md
for file in source/Cheatsheet-0*.md; do
  cat "$file" >> source/Cheatsheet-All.md
  echo -e "\n\n" >> source/Cheatsheet-All.md
done

rm -f cheatsheet.tex cheatsheet.pdf

# 生成tex文件
pandoc "source/Cheatsheet-All.md" \
    --from markdown-simple_tables-multiline_tables-pipe_tables \
    -o cheatsheet.tex \
    --variable=documentclass:extarticle \
    --variable=classoption:8pt \
    --resource-path=. \
    -H template/preamble.tex \
    -B template/before_body.tex \
    -A template/after_body.tex

# # 直接从tex文件编译PDF (使用xelatex以支持中文)
xelatex cheatsheet.tex >> /dev/null
xelatex cheatsheet.tex >> /dev/null  # 运行两次以确保引用和目录正确

# # 清理临时文件
rm -f cheatsheet.aux cheatsheet.log cheatsheet.out source/Cheatsheet-All.md