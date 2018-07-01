DATASET_DIR=/media/yinyuecheng/LENOVO/voc_dataset/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/
OUTPUT_DIR=./tf_records
python3 tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2007_train \
    --output_dir=${OUTPUT_DIR}
