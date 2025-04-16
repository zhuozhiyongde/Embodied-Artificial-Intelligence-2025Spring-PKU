pandoc "02-Robotics-I.md" \
    --from markdown-simple_tables-multiline_tables-pipe_tables \
    -o cheatsheet.pdf \
    --pdf-engine=xelatex \
    --variable=documentclass:extarticle \
    --variable=classoption:8pt \
    -H preamble.tex \
    -B before_body.tex \
    -A after_body.tex \
    --resource-path=.