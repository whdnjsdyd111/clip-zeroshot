batch_size=100

for model in RN50 RN101 RN50x4 RN50x16 RN50x64
do
    for file_name in prompt3.txt prompt4.txt prompt5.txt
    do
        python clip_zero.py \
        --model ${model} \
        --batch_size ${batch_size} \
        --file_name ${file_name}
    done
done