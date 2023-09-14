
for f in *.svg; do
inkscape ${f} --export-pdf=${f%.*}.pdf
done
