python neat_train.py \
       -p "./p/default_neat_mod.json" \
       --no_check_best \
       -n 8 



python neat_test.py \
                -i "./log/test_best.out" \
                -d "./p/smilevolley_neat.json" \
                -p "./p/smilevolley_neat.json" \
                -r 4 \
                -v true