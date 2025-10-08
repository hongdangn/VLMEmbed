mkdir -p vlm2vec_train/MMEB-train/image
wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/A-OKVQA.zip
unzip A-OKVQA.zip -d ./vlm2vec_train/MMEB-train/image/
rm A-OKVQA.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/CIRR.zip
unzip CIRR.zip -d ./vlm2vec_train/MMEB-train/image/
rm CIRR.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/ChartQA.zip
unzip ChartQA.zip -d ./vlm2vec_train/MMEB-train/image/
rm ChartQA.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/DocVQA.zip
unzip DocVQA.zip -d ./vlm2vec_train/MMEB-train/image/
rm DocVQA.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/HatefulMemes.zip
unzip HatefulMemes.zip -d ./vlm2vec_train/MMEB-train/image/
rm HatefulMemes.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/ImageNet_1K.zip
unzip ImageNet_1K.zip -d ./vlm2vec_train/MMEB-train/image/
rm ImageNet_1K.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/InfographicsVQA.zip
unzip InfographicsVQA.zip -d ./vlm2vec_train/MMEB-train/image/
rm InfographicsVQA.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/MSCOCO.zip
unzip MSCOCO.zip -d ./vlm2vec_train/MMEB-train/image/
rm MSCOCO.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/MSCOCO_i2t.zip
unzip MSCOCO_i2t.zip -d ./vlm2vec_train/MMEB-train/image/
rm MSCOCO_i2t.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/MSCOCO_t2i.zip
unzip MSCOCO_t2i.zip -d ./vlm2vec_train/MMEB-train/image/
rm MSCOCO_t2i.zip
