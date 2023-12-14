sudo apt-get update
sudo apt-get install -y curl unzip xvfb libxi6 libgconf-2-4
sudo apt-get install -y libgeos-dev
sudo apt-get install -y default-jdk

cd -

pip install -r requirements.txt