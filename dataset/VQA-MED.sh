mkdir VQA-MED
wget https://zenodo.org/api/records/10499039/files-archive
unzip files-archive
unzip ImageClef-2019-VQA-Med-Training.zip
unzip ImageClef-2019-VQA-Med-Validation.zip
unzip VQAMed2019Test.zip
rm -rf files-archive
rm -rf __MACOSX
rm ImageClef-2019-VQA-Med-Training.zip
rm ImageClef-2019-VQA-Med-Validation.zip
rm VQAMed2019Test.zip
mv ImageClef-2019-VQA-Med-Training VQA-MED
mv ImageClef-2019-VQA-Med-Validation VQA-MED
mv VQAMed2019Test VQA-MED
