# to tex
pandoc "Cheatsheet-03-Policy.md" \
    --from markdown-simple_tables-multiline_tables-pipe_tables \
    -o cheatsheet.tex \
    --variable=documentclass:extarticle \
    --variable=classoption:8pt \
    --resource-path=. \
    -H preamble.tex \
    -B before_body.tex \
    -A after_body.tex

# to pdf
pandoc "Cheatsheet-03-Policy.md" \
    --from markdown-simple_tables-multiline_tables-pipe_tables \
    -o cheatsheet.pdf \
    --pdf-engine=xelatex \
    --variable=documentclass:extarticle \
    --variable=classoption:8pt \
    -H preamble.tex \
    -B before_body.tex \
    -A after_body.tex \
    --resource-path=.