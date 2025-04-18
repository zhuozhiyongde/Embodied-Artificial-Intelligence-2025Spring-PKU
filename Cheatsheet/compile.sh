cat source/Cheatsheet-0*.md > source/Cheatsheet-All.md

# to tex
pandoc "source/Cheatsheet-All.md" \
    --from markdown-simple_tables-multiline_tables-pipe_tables \
    -o cheatsheet.tex \
    --variable=documentclass:extarticle \
    --variable=classoption:8pt \
    --resource-path=. \
    -H template/preamble.tex \
    -B template/before_body.tex \
    -A template/after_body.tex

# to pdf
pandoc "source/Cheatsheet-All.md" \
    --from markdown-simple_tables-multiline_tables-pipe_tables \
    -o cheatsheet.pdf \
    --pdf-engine=xelatex \
    --variable=documentclass:extarticle \
    --variable=classoption:8pt \
    -H template/preamble.tex \
    -B template/before_body.tex \
    -A template/after_body.tex \
    --resource-path=.

rm source/Cheatsheet-All.md