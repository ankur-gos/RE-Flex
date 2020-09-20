python3 -m squad.prepro --mode single --single_path /data/test.json --target_dir inter --glove_dir .
python3 -m basic.cli --mode forward --batch_size 1 --len_opt --cluster --data_dir inter --eval_path inter/eval.pkl.gz --shared_path out/basic/00/shared.json --answer_path pred.json --device-type cpu
mv pred.json /data/
python3 evaluate-v2.0.py /data/test.json /data/pred.json -o /data/eval.json -s
