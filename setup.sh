# Step 1: Build ddprofiler
python3 yamlTemplate.py "$1" "$2"
cd aurum/ddprofiler
chmod +x build.sh
./build.sh
# Step 2: Download and Run Elasticsearch
cd ../.. # Please select the target folder you would like to download
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-6.0.0.tar.gz
tar -xzf "$(pwd)/elasticsearch-6.0.0.tar.gz"
rm "$(pwd)/elasticsearch-6.0.0.tar.gz"
# shellcheck disable=SC2164
ls
# shellcheck disable=SC2164
cd elasticsearch-6.0.0/bin/
./elasticsearch -d # -d flag to run it as a daemon
# Ensure Elasticsearch is up and running before proceeding
sleep 30
# Step 3: Generate YAML file using python script
# Assuming we are back in the initial script directory
cd ../..
cd aurum/ddprofiler
bash run.sh --sources Dataset.yml
# Step 5: Deployment
cd ..
python3 networkbuildercoordinator.py --opath test/testmodel/