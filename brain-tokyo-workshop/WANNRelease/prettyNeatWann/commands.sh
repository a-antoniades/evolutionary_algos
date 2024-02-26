python neat_train.py -p "./p/smilevolley_neat.json" -n 16 


python neat_test.py \
                -i "./log/test_best.out" \
                -d "./p/smilevolley_neat.json" \
                -p "./p/smilevolley_neat.json" \
                -r 4 \
                -v false