mkdir AQUA
wget https://researchdata.aston.ac.uk/380/1/SemArt.zip
unzip SemArt.zip
mv SemArt/Images AQUA
rm SemArt.zip
rm -rf SemArt
git clone https://github.com/noagarcia/ArtVQA
mv ArtVQA/AQUA/*.json AQUA
rm -rf ArtVQA
