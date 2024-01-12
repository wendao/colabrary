if [ ! -d params ]; then
  # get code
  pip -q install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1
  # for debugging
  ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign

  # alphafold params
  mkdir params
  curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C params

  wget -qnc https://zhanggroup.org/TM-score/TMscore.cpp
  g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp
fi

# get data
NAME="1mjc"
wget https://files.ipd.uw.edu/pub/decoyset/natives/{NAME}.pdb
wget https://files.ipd.uw.edu/pub/decoyset/decoys/{NAME}.zip
unzip {NAME}.zip

