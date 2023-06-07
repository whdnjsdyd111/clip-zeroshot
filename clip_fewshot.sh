batch_size=100
epochs=100

for model in RN50 RN50x64
do
    python clip_few.py \
    --model ${model} \
    --batch_size ${batch_size} \
    --epoch ${epochs}
done