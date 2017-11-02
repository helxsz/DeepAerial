python convert_gt.py /home/sizhexi/Documents/data/ISPRS/Vaihingen/gts_for_participants/*.tif --from-color --out /home/sizhexi/Documents/data/ISPRS/Vaihingen/gts_numpy/
python extract_images.py
python create_lmdb.py

apt-get install graphviz
python training.py --niter 40000 --update 1000 --init /home/sizhexi/caffe/py-faster-rcnn/data/imagenet_models/VGG16.v2.caffemodel --snapshot ../snapshots/

python inference_patches.py /home/sizhexi/caffe/DeepNetsForEO/results/33.png   --weights ../models/segnet_vaihingen_128x128_fold1_iter_60000.caffemodel --dir ../results/

python inference.py 1 3 --weights ../models/segnet_vaihingen_128x128_fold1_iter_60000.caffemodel

python inference_patches.py /home/sizhexi/caffe/DeepNetsForEO/data/google/0.png   --weights ../models/segnet_vaihingen_128x128_fold1_iter_60000.caffemodel --dir ../results/

python inference_patches.py /home/sizhexi/Documents/data/ISPRS/Vaihingen/vaihingen_128_128_32_fold1/irrg_test/21.png /home/sizhexi/Documents/data/ISPRS/Vaihingen/vaihingen_128_128_32_fold1/irrg_test/22.png /home/sizhexi/Documents/data/ISPRS/Vaihingen/vaihingen_128_128_32_fold1/irrg_test/23.png --weights ../models/segnet_vaihingen_128x128_fold1_iter_60000.caffemodel --dir ../results/


