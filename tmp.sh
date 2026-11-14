latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex

latexmk -pdf -pvc -view=none -interaction=nonstopmode \
  -halt-on-error -file-line-error main.tex