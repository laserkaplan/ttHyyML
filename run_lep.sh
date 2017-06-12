plot_name=$1
python train.py -c lep -s 5 -n ${plot_name} --save
python applyWeight.py -c l -m models/model_leptonic_shallow.h5 -i ttH -t output -n test
python applyWeight.py -c l -m models/model_leptonic_shallow.h5 -i tHjb_plus1 -t output -n test
python applyWeight.py -c l -m models/model_leptonic_shallow.h5 -i tWH_plus1 -t output -n test
python applyWeight.py -c l -m models/model_leptonic_shallow.h5 -i ggH -t output -n test
python applyWeight.py -c l -m models/model_leptonic_shallow.h5 -i VBF -t output -n test
python applyWeight.py -c l -m models/model_leptonic_shallow.h5 -i WH -t output -n test
python applyWeight.py -c l -m models/model_leptonic_shallow.h5 -i ZH -t output -n test
python applyWeight.py -c l -m models/model_leptonic_shallow.h5 -i data -t output -n test
